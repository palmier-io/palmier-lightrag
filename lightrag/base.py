from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "global"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    top_k: int = 60
    # Number of tokens for the original chunks.
    max_token_for_text_unit: int = 4000
    # Number of tokens for the relationship descriptions
    max_token_for_global_context: int = 4000
    # Number of tokens for the entity descriptions
    max_token_for_local_context: int = 4000


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError
    
    async def delete_by_ids(self, ids: list[str]):
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def get_by_field(
        self, field: str, values: list[str]
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError
    
    async def delete_by_ids(self, ids: list[str]):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError
    
    async def delete_edge(self, source_node_id: str, target_node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")
    
    async def get_nodes_by_property(
        self, 
        property_name: str, 
        property_value: Any,
        split_by_sep: bool = False
    ) -> list[dict]:
        """Get all nodes that have a specific property value.
        
        Args:
            property_name: The name of the property to match
            property_value: The value to match against
            split_by_sep: If True, treats property value as GRAPH_FIELD_SEP-separated 
                         string and matches if value exists in any part
        Returns:
            List of node dictionaries matching the criteria
        """
        raise NotImplementedError

    async def get_edges_by_property(
        self, 
        property_name: str, 
        property_value: Any,
        split_by_sep: bool = False
    ) -> list[dict]:
        """Get all edges that have a specific property value.
        
        Args:
            property_name: The name of the property to match
            property_value: The value to match against
            split_by_sep: If True, treats property value as GRAPH_FIELD_SEP-separated 
                         string and matches if value exists in any part
        Returns:
            List of edge dictionaries matching the criteria
        """
        raise NotImplementedError
    
