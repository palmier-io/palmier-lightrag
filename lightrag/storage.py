import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
    split_string_by_multi_markers,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)

from .prompt import GRAPH_FIELD_SEP


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def get_by_field(self, field: str, values: list) -> dict[str, dict]:
        values_set = set(values)  # Convert to set for O(1) lookup
        return {
            doc_id: doc
            for doc_id, doc in self._data.items()
            if doc.get(field) in values_set
        }

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        for key, new_value in data.items():
            if key in self._data:
                self._data[key].update(new_value)
            else:
                self._data[key] = new_value
        return data

    async def drop(self):
        self._data = {}

    async def delete_by_ids(self, ids: list[str]):
        for id in ids:
            del self._data[id]


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        # Separate data into content changes and metadata-only changes
        content_changes = {}
        metadata_changes = {}

        for k, v in data.items():
            existing = self._client.get([k])
            if not existing or existing[0].get("content") != v.get("content"):
                content_changes[k] = v
            else:
                metadata_changes[k] = v

        results = []

        # Handle metadata-only updates
        if metadata_changes:
            metadata_list = [
                {
                    "__id__": k,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                    "__vector__": self._client.get([k])[0][
                        "__vector__"
                    ],  # Reuse existing embedding
                }
                for k, v in metadata_changes.items()
            ]
            results.extend(self._client.upsert(datas=metadata_list))

        # Handle content changes that need new embeddings
        if content_changes:
            list_data = [
                {
                    "__id__": k,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                }
                for k, v in content_changes.items()
            ]
            contents = [v["content"] for v in content_changes.values()]
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]
            embedding_tasks = [self.embedding_func(batch) for batch in batches]
            embeddings_list = []
            for f in tqdm_async(
                asyncio.as_completed(embedding_tasks),
                total=len(embedding_tasks),
                desc="Generating embeddings",
                unit="batch",
            ):
                embeddings = await f
                embeddings_list.append(embeddings)
            embeddings = np.concatenate(embeddings_list)
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results.extend(self._client.upsert(datas=list_data))
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def delete_by_ids(self, ids: list[str]):
        self._client.delete(ids)

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()

    async def drop(self):
        pass


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def create_index(self):
        pass

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.debug(f"Node {node_id} deleted from the graph.")
        else:
            logger.debug(f"Node {node_id} not found in the graph for deletion.")

    async def delete_edge(self, source_node_id: str, target_node_id: str):
        if self._graph.has_edge(source_node_id, target_node_id):
            self._graph.remove_edge(source_node_id, target_node_id)
            logger.debug(
                f"Edge {source_node_id}->{target_node_id} deleted from the graph."
            )
        else:
            logger.debug(
                f"Edge {source_node_id}->{target_node_id} not found in the graph for deletion."
            )

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    async def get_nodes_by_property(
        self, property_name: str, property_value: Any, split_by_sep: bool = False
    ) -> list[dict]:
        matching_nodes = []
        # Convert property_value to list if it isn't already
        search_values = (
            property_value if isinstance(property_value, list) else [property_value]
        )

        for node, data in self._graph.nodes(data=True):
            prop_value = data.get(property_name)
            if prop_value is None:
                continue

            if split_by_sep:
                # Split stored value by GRAPH_FIELD_SEP
                stored_values = split_string_by_multi_markers(
                    prop_value, [GRAPH_FIELD_SEP]
                )
                # Check if any search value matches any stored value
                if any(search_val in stored_values for search_val in search_values):
                    matching_nodes.append({**data, "id": node})
            else:
                # For non-split values, check if property value matches any search value
                if prop_value in search_values:
                    matching_nodes.append({**data, "id": node})
        return matching_nodes

    async def get_edges_by_property(
        self, property_name: str, property_value: Any, split_by_sep: bool = False
    ) -> list[dict]:
        matching_edges = []
        # Convert property_value to list if it isn't already
        search_values = (
            property_value if isinstance(property_value, list) else [property_value]
        )

        for source, target, data in self._graph.edges(data=True):
            prop_value = data.get(property_name)
            if prop_value is None:
                continue

            if split_by_sep:
                # Split stored value by GRAPH_FIELD_SEP
                stored_values = split_string_by_multi_markers(
                    prop_value, [GRAPH_FIELD_SEP]
                )
                # Check if any search value matches any stored value
                if any(search_val in stored_values for search_val in search_values):
                    matching_edges.append({**data, "source": source, "target": target})
            else:
                # For non-split values, check if property value matches any search value
                if prop_value in search_values:
                    matching_edges.append({**data, "source": source, "target": target})
        return matching_edges

    async def drop(self):
        logger.info("Resetting NetworkX Graph Storage in memory...")
        self._graph = nx.Graph()
        if os.path.exists(self._graphml_xml_file):
            logger.info(
                f"Removing NetworkX Graph Storage file {self._graphml_xml_file}..."
            )
            os.remove(self._graphml_xml_file)
