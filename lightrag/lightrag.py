import asyncio
import os
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
    local_query,
    global_query,
    hybrid_query,
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

from .palmier_storage import (
    SupabaseChunksStorage,
    S3DocsStorage,
    QdrantStorage,
    NeptuneCypherStorage,
)

from .chunking import (
    CodeChunker,
    get_language_from_file,
    traverse_directory,
    FILES_TO_IGNORE,
    SUPPORT_LANGUAGES,
)

from .kg.neo4j_impl import Neo4JStorage

from .kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class LightRAG:
    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    environment: str = field(default="dev")

    # storage
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
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
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
        log_file = os.path.join(self.working_dir, "lightrag.log")
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
            # embedding_func=self.embedding_func,
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
            "SupabaseChunksStorage": SupabaseChunksStorage,
            "S3DocsStorage": S3DocsStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "QdrantStorage": QdrantStorage,
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

    async def ainsert_files(self, directory: str, file_paths: Optional[list[str]] = None):
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
                logger.info(f"[Traversing] No file provided to ainsert_files, traversing {directory}")
                file_paths = traverse_directory(directory)

            # Create a new document for each file
            logger.info(f"[Filtering] processing {len(file_paths)} files at {directory}")
            new_docs = {}
            for full_file_path in file_paths:

                # Filter out unwanted/unsupported files
                if any(full_file_path.endswith(ext) for ext in FILES_TO_IGNORE):
                    continue
                language = get_language_from_file(full_file_path)
                if language != "text only" and language not in SUPPORT_LANGUAGES:
                    continue

                with open(full_file_path, "r") as f:
                    content = f.read()

                relative_file_path = os.path.relpath(full_file_path, directory)
                # use hash(file_path) as doc_id
                new_docs[compute_mdhash_id(relative_file_path.strip(), prefix="doc-")] = {
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
            for doc_key, doc in new_docs.items():
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
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "hybrid":
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
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
