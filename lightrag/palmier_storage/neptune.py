import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import aiohttp
from lightrag.utils import logger
from ..base import BaseGraphStorage
from ..prompt import GRAPH_FIELD_SEP

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class NeptuneCypherStorage(BaseGraphStorage):
    node_label: str = "Entity"

    def __init__(self, namespace, global_config):
        super().__init__(namespace=namespace, global_config=global_config)
        self._session = None

        NEPTUNE_ENDPOINT = os.environ["NEPTUNE_ENDPOINT"]
        NEPTUNE_PORT = os.environ.get("NEPTUNE_PORT", "8182")
        self.neptune_url = f"https://{NEPTUNE_ENDPOINT}:{NEPTUNE_PORT}/openCypher"

        storage_param = global_config.get("storage_params", {})
        if not storage_param:
            raise ValueError("storage parameter is required in global config")

        self.repository = storage_param.get("repository")
        self.repository_id = storage_param.get("repository_id")
        if not self.repository or not self.repository_id:
            raise ValueError(
                "repository and repository_id are required in storage parameter"
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def create_index(self):
        # Index creation in Amazon Neptune for openCypher is not directly supported.
        # Neptune automatically indexes certain properties.
        pass

    def db_retry_decorator():
        """
        Retry decorator for database operations.
        """
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(
                (
                    aiohttp.ClientError,
                    Exception,  # other exceptions can be added here
                )
            ),
        )

    async def _get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _execute_cypher_query(self, query: str, params: Dict = None):
        url = self.neptune_url
        headers = {"Content-Type": "application/json"}
        data = {"query": query}
        if params:
            data["parameters"] = params

        session = await self._get_session()
        async with session.post(url, json=data, headers=headers, ssl=False) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Error executing query: {text}")
                response.raise_for_status()
            result = await response.json()
            return result

    @db_retry_decorator()
    async def has_node(self, node_id: str) -> bool:
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            RETURN COUNT(n) AS node_count
        """
        params = {"repository_id": self.repository_id, "node_id": node_id}
        try:
            result = await self._execute_cypher_query(query, params)
            node_count = result["results"][0]["data"][0]["row"][0]
            return node_count > 0
        except Exception as e:
            logger.error(f"Error in has_node: {str(e)}")
            raise

    @db_retry_decorator()
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        source_node_id = source_node_id.strip('"')
        target_node_id = target_node_id.strip('"')
        query = f"""
            MATCH (a:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            -[r {{repository_id: $repository_id}}]->
            (b:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
            RETURN COUNT(r) AS edge_count
        """
        params = {
            "repository_id": self.repository_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        try:
            result = await self._execute_cypher_query(query, params)
            edge_count = result["results"][0]["data"][0]["row"][0]
            return edge_count > 0
        except Exception as e:
            logger.error(f"Error in has_edge: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_node(self, node_id: str) -> Union[dict, None]:
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            RETURN n
            LIMIT 1
        """
        params = {"repository_id": self.repository_id, "node_id": node_id}
        try:
            result = await self._execute_cypher_query(query, params)
            data = result["results"][0]["data"]
            if data:
                node_props = data[0]["row"][0]
                logger.debug(f"get_node: result: {node_props}")
                return node_props
            return None
        except Exception as e:
            logger.error(f"Error in get_node: {str(e)}")
            raise

    @db_retry_decorator()
    async def node_degree(self, node_id: str) -> int:
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            RETURN SIZE((n)--()) AS totalEdgeCount
        """
        params = {"repository_id": self.repository_id, "node_id": node_id}
        try:
            result = await self._execute_cypher_query(query, params)
            edge_count = result["results"][0]["data"][0]["row"][0]
            logger.debug(f"node_degree: result: {edge_count}")
            return edge_count
        except Exception as e:
            logger.error(f"Error in node_degree: {str(e)}")
            raise

    @db_retry_decorator()
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(f"edge_degree: result: {degrees}")
        return degrees

    @db_retry_decorator()
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        source_node_id = source_node_id.strip('"')
        target_node_id = target_node_id.strip('"')
        query = f"""
            MATCH (start:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            -[r {{repository_id: $repository_id}}]->
            (end:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
            RETURN r
            LIMIT 1
        """
        params = {
            "repository_id": self.repository_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        try:
            result = await self._execute_cypher_query(query, params)
            data = result["results"][0]["data"]
            if data:
                edge_props = data[0]["row"][0]
                logger.debug(f"get_edge: result: {edge_props}")
                return edge_props
            return None
        except Exception as e:
            logger.error(f"Error in get_edge: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        source_node_id = source_node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            -[r]->(connected:{self.node_label} {{repository_id: $repository_id}})
            RETURN n.node_id AS source_node_id, connected.node_id AS connected_node_id
        """
        params = {"repository_id": self.repository_id, "source_node_id": source_node_id}
        try:
            result = await self._execute_cypher_query(query, params)
            data = result["results"][0]["data"]
            edges = []
            for record in data:
                source_id = record["row"][0]
                connected_id = record["row"][1]
                if source_id and connected_id:
                    edges.append((source_id, connected_id))
            logger.debug(f"get_node_edges: result: {edges}")
            return edges
        except Exception as e:
            logger.error(f"Error in get_node_edges: {str(e)}")
            raise

    @db_retry_decorator()
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        node_id = node_id.strip('"')
        properties = {
            **node_data,
            "repository_id": self.repository_id,
            "node_id": node_id,
        }

        # Build the Cypher MERGE query
        props_str = ", ".join(f"{key}: ${key}" for key in properties.keys())
        query = f"""
            MERGE (n:{self.node_label} {{node_id: $node_id, repository_id: $repository_id}})
            SET n += {{{props_str}}}
        """
        params = properties
        try:
            await self._execute_cypher_query(query, params)
            logger.debug(
                f"Upserted node with node_id '{node_id}' in repository '{self.repository_id}' and properties: {properties}"
            )
        except Exception as e:
            logger.error(f"Error during upsert_node: {str(e)}")
            raise

    @db_retry_decorator()
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        source_node_id = source_node_id.strip('"')
        target_node_id = target_node_id.strip('"')
        relationship_type = "DIRECTED"
        edge_properties = {**edge_data, "repository_id": self.repository_id}

        # Build the Cypher MERGE query for edge
        props_str = ", ".join(f"{key}: ${key}" for key in edge_properties.keys())
        query = f"""
            MATCH (source:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            MATCH (target:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
            MERGE (source)-[r:{relationship_type} {{repository_id: $repository_id}}]->(target)
            SET r += {{{props_str}}}
        """
        params = {
            "repository_id": self.repository_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            **edge_properties,
        }
        try:
            await self._execute_cypher_query(query, params)
            logger.debug(
                f"Upserted edge from '{source_node_id}' to '{target_node_id}' with properties: {edge_properties}"
            )
        except Exception as e:
            logger.error(f"Error during upsert_edge: {str(e)}")
            raise

    @db_retry_decorator()
    async def delete_node(self, node_id: str):
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            DETACH DELETE n
        """
        params = {"repository_id": self.repository_id, "node_id": node_id}
        try:
            await self._execute_cypher_query(query, params)
            logger.debug(f"Node {node_id} deleted from the graph.")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @db_retry_decorator()
    async def delete_edge(self, source_node_id: str, target_node_id: str):
        source_node_id = source_node_id.strip('"')
        target_node_id = target_node_id.strip('"')
        relationship_type = "DIRECTED"
        query = f"""
            MATCH (source:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            -[r:{relationship_type} {{repository_id: $repository_id}}]->
            (target:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
            DELETE r
        """
        params = {
            "repository_id": self.repository_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        try:
            await self._execute_cypher_query(query, params)
            logger.debug(
                f"Edge {source_node_id}->{target_node_id} deleted from the graph."
            )
        except Exception as e:
            logger.error(f"Error during edge deletion: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_nodes_by_property(
        self, property_name: str, property_value: Any, split_by_sep: bool = False
    ) -> List[Dict]:
        if split_by_sep:
            query = f"""
                MATCH (n:{self.node_label} {{repository_id: $repository_id}})
                WHERE n.{property_name} IS NOT NULL
                AND ANY(x IN split(n.{property_name}, $sep) WHERE x = $property_value)
                RETURN n
            """
            params = {
                "repository_id": self.repository_id,
                "property_value": property_value,
                "sep": GRAPH_FIELD_SEP,
            }
        else:
            query = f"""
                MATCH (n:{self.node_label} {{repository_id: $repository_id}})
                WHERE n.{property_name} = $property_value
                RETURN n
            """
            params = {
                "repository_id": self.repository_id,
                "property_value": property_value,
            }
        try:
            result = await self._execute_cypher_query(query, params)
            data = result["results"][0]["data"]
            nodes = [record["row"][0] for record in data]
            logger.debug(f"get_nodes_by_property: result: {nodes}")
            return nodes
        except Exception as e:
            logger.error(f"Error in get_nodes_by_property: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_edges_by_property(
        self, property_name: str, property_value: Any, split_by_sep: bool = False
    ) -> List[Dict]:
        relationship_type = "DIRECTED"
        if split_by_sep:
            query = f"""
                MATCH ()-[r:{relationship_type} {{repository_id: $repository_id}}]-()
                WHERE r.{property_name} IS NOT NULL
                AND ANY(x IN split(r.{property_name}, $sep) WHERE x = $property_value)
                RETURN r
            """
            params = {
                "repository_id": self.repository_id,
                "property_value": property_value,
                "sep": GRAPH_FIELD_SEP,
            }
        else:
            query = f"""
                MATCH ()-[r:{relationship_type} {{repository_id: $repository_id}}]-()
                WHERE r.{property_name} = $property_value
                RETURN r
            """
            params = {
                "repository_id": self.repository_id,
                "property_value": property_value,
            }
        try:
            result = await self._execute_cypher_query(query, params)
            data = result["results"][0]["data"]
            edges = [record["row"][0] for record in data]
            logger.debug(f"get_edges_by_property: result: {edges}")
            return edges
        except Exception as e:
            logger.error(f"Error in get_edges_by_property: {str(e)}")
            raise

    @db_retry_decorator()
    async def drop(self):
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id}})
            DETACH DELETE n
        """
        params = {"repository_id": self.repository_id}
        try:
            await self._execute_cypher_query(query, params)
            logger.info(
                f"All nodes and edges for repository_id '{self.repository_id}' have been deleted."
            )
            await self.close()
            logger.info(
                f"Neptune connection closed after cleaning up repository_id '{self.repository_id}'."
            )
        except Exception as e:
            logger.error(
                f"Error during drop operation for repository_id '{self.repository_id}': {str(e)}"
            )
            raise
