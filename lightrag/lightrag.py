import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Optional
import shutil
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    delete_by_chunk_ids,
    kg_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from .chunking import (
    CodeChunker,
    get_language_from_file,
    traverse_directory,
    should_ignore_file,
)

from .palmier import (
    update_summary,
)

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import an external module and return a class from it."""

    def import_class(**kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name)

        # Get the class from the module
        cls = getattr(module, class_name)

        # Return an instance if kwargs are provided, otherwise return the class
        return cls(**kwargs) if kwargs else cls

    # Return the import_class function itself, not its result
    return import_class


Neo4JStorage = lazy_external_import("lightrag.kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import("lightrag.kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(
    "lightrag.kg.oracle_impl", "OracleGraphStorage"
)
OracleVectorDBStorage = lazy_external_import(
    "lightrag.kg.oracle_impl", "OracleVectorDBStorage"
)
MilvusVectorDBStorge = lazy_external_import(
    "lightrag.kg.milvus_impl", "MilvusVectorDBStorge"
)
MongoKVStorage = lazy_external_import("lightrag.kg.mongo_impl", "MongoKVStorage")
SupabaseChunksStorage = lazy_external_import(
    "lightrag.kg.supabase_impl", "SupabaseChunksStorage"
)
S3DocsStorage = lazy_external_import("lightrag.kg.s3_impl", "S3DocsStorage")
QdrantStorage = lazy_external_import("lightrag.kg.qdrant_impl", "QdrantStorage")
NeptuneCypherStorage = lazy_external_import(
    "lightrag.kg.neptune_impl", "NeptuneCypherStorage"
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class LightRAG:
    environment: str = field(default="dev")

    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    docs_storage: str = field(default="JsonKVStorage")
    chunks_storage: str = field(default="JsonKVStorage")
    llm_response_cache_storage: str = field(default="JsonKVStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 800
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    chunk_summary_enabled: bool = False

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    llm_model_name: str = (
        "gpt-4o-mini"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    )
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    storage_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join("lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Initialize storage classes with parameters from storage_params
        storage_classes = self._get_storage_class()

        # Helper function to get storage class with params
        def get_storage_with_params(storage_name: str) -> Type:
            storage_cls = storage_classes[storage_name]
            storage_params = self.storage_params.get(storage_name, {})
            return partial(storage_cls, **storage_params)

        # Initialize all storage classes with their specific params
        self.docs_storage_cls = get_storage_with_params(self.docs_storage)
        self.chunks_storage_cls = get_storage_with_params(self.chunks_storage)
        self.llm_response_cache_storage_cls = get_storage_with_params(
            self.llm_response_cache_storage
        )
        self.vector_db_storage_cls = get_storage_with_params(self.vector_storage)
        self.graph_storage_cls = get_storage_with_params(self.graph_storage)

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.llm_response_cache_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.docs_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.chunks_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.summaries_vdb = self.vector_db_storage_cls(
            namespace="summaries",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"file_path", "type"},
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "SupabaseChunksStorage": SupabaseChunksStorage,
            "S3DocsStorage": S3DocsStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "QdrantStorage": QdrantStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "NeptuneCypherStorage": NeptuneCypherStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert_files(self, directory: str, file_paths: Optional[list[str]] = None):
        """
        Palmier Specific - inserting file(s) to the knowledge graph

        Input:
        - directory: the directory where the github repository is downloaded to
        - file_paths [optional]: a list of full file paths to insert. If not provided, the function will traverse the directory and insert all files.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_files(directory, file_paths))

    async def ainsert_files(
        self, directory: str, file_paths: Optional[list[str]] = None
    ):
        """Palmier Specific - inserting file(s) to the knowledge graph"""
        update_storage = False
        try:
            code_chunker = CodeChunker(
                directory,
                target_tokens=self.chunk_token_size,
                overlap_token_size=self.chunk_overlap_token_size,
                tiktoken_model=self.tiktoken_model_name,
                summary_enabled=self.chunk_summary_enabled,
            )

            if file_paths is None:
                logger.info(
                    f"[Traversing] No file provided to ainsert_files, traversing {directory}"
                )
                file_paths = traverse_directory(directory)

            logger.info("[Updating Summary] Updating the summary tree")
            await update_summary(
                directory, file_paths, self.summaries_vdb, self.llm_model_func
            )

            # Create a new document for each file
            new_docs = {}
            for full_file_path in file_paths:
                relative_file_path = os.path.relpath(full_file_path, directory)
                if should_ignore_file(relative_file_path):
                    continue
                logger.debug("Reading file: ", full_file_path)
                with open(full_file_path, "r") as f:
                    content = f.read()
                language = get_language_from_file(full_file_path)
                # use hash(file_path) as doc_id
                new_docs[
                    compute_mdhash_id(relative_file_path.strip(), prefix="doc-")
                ] = {
                    "file_path": relative_file_path,
                    "language": language,
                    "content": content,
                }

            update_storage = True
            logger.info(f"[New Docs] upserting {len(new_docs)} docs")
            await self.full_docs.upsert(new_docs)

            # Chunking by either tree-sitter or token size
            logger.info(f"[Chunking] chunking {len(new_docs)} docs")
            new_chunks = {}
            for doc_key, doc in new_docs.items():
                logger.debug(f"Chunking file {doc['file_path']}")
                chunks = {
                    # use hash(content) as chunk_id
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in code_chunker.chunk_file(doc["file_path"], doc["content"])
                }
                new_chunks.update(chunks)

            # Get all exsiting chunks belonging to new docs
            previous_chunks = await self.text_chunks.get_by_field(
                "full_doc_id", list(new_docs.keys())
            )

            # Filter out chunks that have new content
            _add_chunks_keys = set(new_chunks.keys()) - set(previous_chunks.keys())

            # Filter out outdated chunks
            _remove_chunks_keys = set(previous_chunks.keys()) - set(new_chunks.keys())

            # Filter out chunks that only need metadata update (no content change)
            _update_chunks_keys = set(previous_chunks.keys()) - set(_remove_chunks_keys)

            adding_chunks = {
                k: v for k, v in new_chunks.items() if k in _add_chunks_keys
            }

            updating_chunks = {
                k: v for k, v in new_chunks.items() if k in _update_chunks_keys
            }

            if not len(_add_chunks_keys):
                logger.warning("All chunks are already in the storage")
                return
            else:
                logger.info(f"[New Chunks] inserting {len(adding_chunks)} chunks")
                await self.chunks_vdb.upsert(adding_chunks)
                await self.text_chunks.upsert(adding_chunks)

            logger.info("[Entity Extraction]...")
            await self.chunk_entity_relation_graph.create_index()
            maybe_new_kg = await extract_entities(
                adding_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            if len(updating_chunks) > 0:
                logger.info(
                    f"[Update Chunks] updating {len(updating_chunks)} chunk metadata"
                )
                await self.chunks_vdb.upsert(updating_chunks)
                await self.text_chunks.upsert(updating_chunks)

            if len(_remove_chunks_keys) > 0:
                logger.info(
                    f"[Remove Chunks] removing {len(_remove_chunks_keys)} outdated chunks"
                )
                await delete_by_chunk_ids(
                    list(_remove_chunks_keys),
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunks_vdb,
                    self.text_chunks,
                )

        finally:
            if update_storage:
                await self._insert_done()

    def delete_files(self, directory: str, file_paths: list[str]):
        """
        Palmier Specific - deleting file(s) from the knowledge graph

        Input:
        - directory: the directory where the github repository is downloaded to
        - file_paths: a list of full file paths to delete
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_files(directory, file_paths))

    async def adelete_files(self, directory: str, file_paths: list[str]):
        update_storage = False
        try:
            relative_file_paths = [
                file_path.replace(directory, "") for file_path in file_paths
            ]
            all_docs = await self.full_docs.get_by_field(
                "file_path", relative_file_paths
            )

            if not len(all_docs):
                logger.warning("Docs are not found in the storage")
                return

            update_storage = True

            all_chunks = await self.text_chunks.get_by_field(
                "full_doc_id", list(all_docs.keys())
            )

            # Delete chunks
            logger.info(f"[Remove Chunks] removing {len(all_chunks)} chunks")
            await delete_by_chunk_ids(
                list(all_chunks.keys()),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
            )

            # Delete docs
            logger.info(f"[Remove Docs] removing {len(all_docs)} docs")
            await self.full_docs.delete_by_ids(list(all_docs.keys()))
        finally:
            if update_storage:
                await self._delete_done()

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.summaries_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            if self.relationships_vdb is not None:
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
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.summaries_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _delete_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.chunks_vdb,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def drop(self):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adrop())

    async def adrop(self):
        # TODO: drop all the storage
        try:
            logger.info("Dropping Graph Storage...")
            await self.chunk_entity_relation_graph.drop()

            logger.info("Dropping Entities Vector Storage...")
            await self.entities_vdb.drop()

            logger.info("Dropping Relationships Vector Storage...")
            await self.relationships_vdb.drop()

            logger.info("Dropping Chunks Vector Storage...")
            await self.chunks_vdb.drop()

            logger.info("Dropping Text Chunks Storage...")
            await self.text_chunks.drop()

            logger.info("Dropping Full Docs Storage...")
            await self.full_docs.drop()

            if os.path.exists(self.working_dir):
                logger.info(f"Removing working directory {self.working_dir}...")
                shutil.rmtree(self.working_dir)

        except Exception as e:
            logger.error(f"Error while dropping storage: {e}")
