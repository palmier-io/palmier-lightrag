import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union, Optional
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    csv_string_to_list,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    QueryResult,
)
from .llm import voyageai_rerank
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=800, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    source_descriptions = {}
    already_entitiy_types = []

    # Get existing node data if it exists
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        # Split and recreate the mapping from existing data
        existing_sources = split_string_by_multi_markers(
            already_node["source_id"], [GRAPH_FIELD_SEP]
        )
        existing_descriptions = split_string_by_multi_markers(
            already_node["description"], [GRAPH_FIELD_SEP]
        )
        for src, desc in zip(existing_sources, existing_descriptions):
            source_descriptions[src] = desc

    # Add new data to the mapping
    for dp in nodes_data:
        source_descriptions[dp["source_id"]] = dp["description"]
        already_entitiy_types.append(dp["entity_type"])

    # Get most common entity type
    entity_type = sorted(
        Counter(already_entitiy_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # Join source_ids and descriptions while maintaining the mapping
    source_id = GRAPH_FIELD_SEP.join(source_descriptions.keys())
    description = GRAPH_FIELD_SEP.join(source_descriptions.values())

    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )

    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    summaries: Optional[dict] = None,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join([f"{k}:{v}" for k,v in PROMPTS["DEFAULT_ENTITY_TYPES"].items()]),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        summary_id = chunk_dp["full_doc_id"].replace("doc-", "sum-")
        file_summary = (
            summaries[summary_id]["content"]
            if summaries and summary_id in summaries
            else ""
        )
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}", file_summary="{file_summary}"
        ).format(**context_base, input_text=content, file_summary=file_summary)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def delete_by_chunk_ids(
    chunk_ids: list[str],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    # Remove chunks from vector db and text chunks
    await chunks_vdb.delete_by_ids(chunk_ids)
    await text_chunks_db.delete_by_ids(chunk_ids)

    # Find all nodes and edges that are connected to the chunks
    all_related_nodes = await knowledge_graph_inst.get_nodes_by_property(
        "source_id", chunk_ids, split_by_sep=True
    )
    all_related_edges = await knowledge_graph_inst.get_edges_by_property(
        "source_id", chunk_ids, split_by_sep=True
    )

    # Update nodes
    deleted_node_count = 0
    for node in all_related_nodes:
        node_name = node.get("id", "")

        # Recreate the mapping
        source_descriptions = {}
        sources = split_string_by_multi_markers(
            node.get("source_id", ""), [GRAPH_FIELD_SEP]
        )
        descriptions = split_string_by_multi_markers(
            node.get("description", ""), [GRAPH_FIELD_SEP]
        )

        for src, desc in zip(sources, descriptions):
            if src not in chunk_ids:  # Only keep sources we're not deleting
                source_descriptions[src] = desc

        if source_descriptions:  # If there are remaining source_ids, update the node
            node["source_id"] = GRAPH_FIELD_SEP.join(source_descriptions.keys())
            node["description"] = GRAPH_FIELD_SEP.join(source_descriptions.values())
            await knowledge_graph_inst.upsert_node(node_name, node)
        else:  # If no source_ids remain, delete the node
            await knowledge_graph_inst.delete_node(node_name)
            await entities_vdb.delete_entity(node_name)
            await relationships_vdb.delete_relation(node_name)
            deleted_node_count += 1

    # Update edges
    deleted_edge_count = 0
    for edge in all_related_edges:
        source_ids = split_string_by_multi_markers(
            edge.get("source_id", ""), [GRAPH_FIELD_SEP]
        )

        # Filter out chunk_ids from source_ids
        filtered_source_ids = [sid for sid in source_ids if sid not in chunk_ids]

        if filtered_source_ids:  # If there are remaining source_ids, update the edge
            edge["source_id"] = GRAPH_FIELD_SEP.join(filtered_source_ids)
            await knowledge_graph_inst.upsert_edge(edge["source"], edge["target"], edge)
        else:  # If no source_ids remain, delete the edge
            await knowledge_graph_inst.delete_edge(edge["source"], edge["target"])
            deleted_edge_count += 1

    logger.info(f"Deleted {deleted_node_count} nodes and {deleted_edge_count} edges")


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    summaries_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    full_docs: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> QueryResult:
    if query.strip() == "":
        return QueryResult(answer=PROMPTS["fail_response"])
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return QueryResult(answer=cached_response)

    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # Set mode
    if query_param.mode not in ["local", "global", "hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return QueryResult(answer=PROMPTS["fail_response"])

    summary_csv = await _get_summaries_from_query(
        query, summaries_vdb, query_param, global_config
    )

    # LLM generate keywords
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(
        query=query,
        examples=examples,
        language=language,
        summary_context=summary_csv,
    )
    logger.info("Analyzing query and generating search parameters")
    reasoning_result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info("kw_prompt result:")
    print(reasoning_result)
    try:
        # json_text = locate_json_string_body_from_string(result) # handled in use_model_func
        match = re.search(r"\{.*\}", reasoning_result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])

        else:
            logger.error("No JSON-like structure found in the result.")
            return QueryResult(answer=PROMPTS["fail_response"])

    # Handle parsing error
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return QueryResult(answer=PROMPTS["fail_response"])

    # Handdle keywords missing
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return QueryResult(answer=PROMPTS["fail_response"])
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty")
        return QueryResult(answer=PROMPTS["fail_response"])
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty")
        return QueryResult(answer=PROMPTS["fail_response"])
    else:
        hl_keywords = ", ".join(hl_keywords)

    # Build context
    keywords = [ll_keywords, hl_keywords]
    thought_process = f"User query: {query} \nThought process: {keywords_data.get('thought_process', [])}"

    chunks_csv = await _get_chunks_from_keywords(
        keywords_data, text_chunks_db, chunks_vdb, query_param
    )

    context = await _build_query_context(
        thought_process,
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        full_docs,
        query_param,
        summary_csv,
        chunks_csv,
        global_config,
    )
    reasoning = dict(keywords_data) if query_param.include_reasoning and keywords_data else None
    if query_param.only_need_context:
        return QueryResult(context=context, reasoning=reasoning)
    if context is None:
        return QueryResult(answer=PROMPTS["fail_response"])
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        repository_name=global_config.get("repository_name"),
    )
    if query_param.only_need_prompt:
        return QueryResult(answer=sys_prompt, reasoning=reasoning)
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return QueryResult(answer=response, reasoning=reasoning)

async def _rerank_context(context: str, query_str: str, query_param: QueryParam, rerank_model: str):
    """
    Rerank the context using voyage_rerank while preserving metadata.
    It compares the content (sources, summaries) or description (entities, relationships) with the user query + thought_process of the LLM.
    """
    if not context.strip():
        return context
        
    # Parse CSV context into list of lists
    context_list = csv_string_to_list(context)
    if len(context_list) <= 1:  # Only header or empty
        return context
        
    header = context_list[0]
    rows = context_list[1:]
    
    # Extract content column index based on header
    content_idx = None
    for i, col in enumerate(header):
        if col.lower() == "content" or col.lower() == "description":
            content_idx = i
            break
    
    if content_idx is None:
        return context  # No content/description column found
        
    # Extract documents for reranking
    documents = [row[content_idx] for row in rows]    
    
    # Get reranked results with scores
    reranked = await voyageai_rerank(query_str, documents, model=rerank_model, top_k=query_param.rerank_top_k)
    
    # Create mapping of content to original row to preserve metadata
    content_to_row = {row[content_idx]: row for row in rows}
    
    # Rebuild rows in reranked order with original metadata
    reranked_rows = []
    for result in reranked:  # result is a RerankingResult
        original_row = content_to_row[result.document]  # Use .document to access the field
        reranked_rows.append(original_row)
    
    # Reconstruct CSV with header and reranked rows
    reranked_context = list_of_list_to_csv([header] + reranked_rows)
    return reranked_context

async def _get_summaries_from_query(
    query: str,
    summaries_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Based on the query, retrieve top k summaries from the vector database"""
    query_chunks = chunking_by_token_size(
        query,
        overlap_token_size=global_config["chunk_overlap_token_size"],
        max_token_size=8192,  # limit for text-embedding-3-small
        tiktoken_model=global_config["tiktoken_model_name"],
    )

    summaries = await asyncio.gather(
        *[
            summaries_vdb.query(chunk["content"], top_k=query_param.top_k)
            for chunk in query_chunks
        ]
    )
    summaries = [item for sublist in summaries for item in sublist]
    summary_context = [["id", "level", "file_path", "score", "content"]]
    seen_file_paths = set()

    for i, s in enumerate(summaries):
        file_path = s["file_path"]
        if file_path in seen_file_paths:
            continue
        seen_file_paths.add(file_path)
        summary_context.append(
            [
                i,
                s["type"],
                file_path,
                s["score"],
                s["content"],
            ]
        )
    logger.info(f"Retrieved {len(summaries)} relevant summaries")
    return list_of_list_to_csv(summary_context)


async def _get_chunks_from_keywords(
    keywords_data: dict,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
) -> str:
    """
    Get chunks using different search parameters and methods:
    - file_path: Given a list of file paths, get all chunks that match any of the file paths
    - refined_queries: Given a list of refined queries, query the vector database for chunks
    - symbol_names: Given a list of symbol names, get all chunks that match any of the symbol names
    """
    file_paths = keywords_data.get("file_paths", [])
    refined_queries = keywords_data.get("refined_queries", [])
    # symbol_names = keywords_data.get("symbol_names", [])

    chunks_vector = await asyncio.gather(
        *[
            chunks_vdb.query(refined_query, top_k=query_param.top_k)
            for refined_query in refined_queries
        ]
    )
    chunks_vector = [item for sublist in chunks_vector for item in sublist]
    chunks_list = [["id", "content", "file_path", "start_line", "end_line"]]
    chunks = await asyncio.gather(
        *[text_chunks_db.get_by_id(t["id"]) for t in chunks_vector]
    )
    logger.info(
        f"Chunk retrieval using refined queries uses {len(chunks_vector)} text units"
    )

    chunks_list.extend(
        [
            i,
            chunk["content"],
            chunk["file_path"],
            chunk["start"]["line"],
            chunk["end"]["line"],
        ]
        for i, chunk in enumerate(chunks)
    )

    file_chunks = await text_chunks_db.get_by_field("file_path", file_paths)
    chunks_list.extend(
        [
            i,
            chunk["content"],
            chunk["file_path"],
            chunk["start"]["line"],
            chunk["end"]["line"],
        ]
        for i, chunk in enumerate(file_chunks)
    )
    logger.info(f"File chunk query uses {len(file_chunks)} text units")

    chunks_csv = list_of_list_to_csv(chunks_list)
    return chunks_csv


async def _build_query_context(
    query: str,
    keywords: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    full_docs: BaseKVStorage,
    query_param: QueryParam,
    summary_csv: str,
    chunks_csv: str,
    global_config: dict,
):
    ll_keywords, hl_keywords = keywords[0], keywords[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_keywords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "Low Level context is None. Return empty Low entity/relationship/source"
            )
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywords == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "High Level context is None. Return empty High entity/relationship/source"
            )
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context, chunks_csv],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )

    if query_param.include_full_file:
        text_units_context_list = csv_string_to_list(text_units_context)[1:]
        file_paths = set([s[2] for s in text_units_context_list])
        docs = await full_docs.get_by_field("file_path", list(file_paths))
        docs_list = [["id", "file_path", "content"]]
        docs_list.extend(
            [[i, d["file_path"], d["content"]] for i, d in enumerate(docs)]
        )
        text_units_context = list_of_list_to_csv(docs_list)
        logger.info(
            f"Converted {len(text_units_context_list)} text units to {len(docs)} full files"
        )

    if query_param.rerank_enabled:
        rerank_model = global_config["rerank_model"]
        logger.info(f"Reranking with query: {query}")
        logger.info(f"Reranking summaries to {query_param.rerank_top_k} results")
        summary_csv = await _rerank_context(summary_csv, query, query_param, rerank_model)
        logger.info(f"Reranking entities to {query_param.rerank_top_k} results")
        entities_context = await _rerank_context(entities_context, query, query_param, rerank_model)
        logger.info(f"Reranking relationships to {query_param.rerank_top_k} results")
        relations_context = await _rerank_context(relations_context, query, query_param, rerank_model)
        logger.info(f"Reranking text units to {query_param.rerank_top_k} results")
        text_units_context = await _rerank_context(text_units_context, query, query_param, rerank_model)

    return f"""
-----Summaries-----
```csv
{summary_csv}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> tuple[str, str, str]:
    # get similar entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content", "file_path", "start_line", "end_line"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append(
            [i, t["content"], t["file_path"], t["start"]["line"], t["end"]["line"]]
        )
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> tuple[str, str, str]:
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content", "file_path", "start_line", "end_line"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append(
            [i, t["content"], t["file_path"], t["start"]["line"], t["end"]["line"]]
        )
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources, naive_sources = sources[0], sources[1], sources[2]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = process_combine_contexts(combined_sources, naive_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> QueryResult:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return QueryResult(answer=cached_response)

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return QueryResult(answer=PROMPTS["fail_response"])

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return QueryResult(answer=PROMPTS["fail_response"])

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return QueryResult(answer=PROMPTS["fail_response"])

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return QueryResult(context=section)

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )

    if query_param.only_need_prompt:
        return QueryResult(answer=sys_prompt)

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )

    return QueryResult(answer=response)
