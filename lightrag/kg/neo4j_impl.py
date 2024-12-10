import asyncio
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import inspect
from lightrag.utils import logger
from ..base import BaseGraphStorage
from ..prompt import GRAPH_FIELD_SEP
from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class Neo4JStorage(BaseGraphStorage):
    node_label: str = "Entity"

    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        URI = os.environ["NEO4J_URI"]
        USERNAME = os.environ["NEO4J_USERNAME"]
        PASSWORD = os.environ["NEO4J_PASSWORD"]

        # Initialize driver without binding to a specific loop
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            URI, auth=(USERNAME, PASSWORD)
        )

        storage_param = global_config.get("storage_params", {})
        if not storage_param:
            raise ValueError("storage parameter is required in global config")

        self.repository = storage_param.get("repository")
        self.repository_id = str(storage_param.get("repository_id"))
        if not self.repository or not self.repository_id:
            raise ValueError(
                "repository and repository_id are required in storage parameter"
            )
        return None

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def create_index(self):
        # Composite Index on repository_id and node_id
        unique_node_query = f"""
        CREATE CONSTRAINT unique_node IF NOT EXISTS
            FOR (n:{self.node_label})
            REQUIRE (n.repository_id, n.node_id) IS UNIQUE
        """

        # Index on repository_id for quick tenant-specific queries
        repository_id_index_query = f"""
        CREATE INDEX repository_id_index IF NOT EXISTS
            FOR (n:{self.node_label})
            ON (n.repository_id)
        """

        async with self._driver.session() as session:
            await session.run(unique_node_query)
            await session.run(repository_id_index_query)

    def db_retry_decorator():
        """
        Retry decorator for database operations.
        """
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(
                (
                    neo4jExceptions.ServiceUnavailable,
                    neo4jExceptions.TransientError,
                    neo4jExceptions.WriteServiceUnavailable,
                    neo4jExceptions.ClientError,
                )
            ),
        )

    @db_retry_decorator()
    async def has_node(self, node_id: str) -> bool:
        try:
            node_id = node_id.strip('"')
            query = f"""
                MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
                RETURN count(n) > 0 AS node_exists
            """
            async with self._driver.session() as session:
                result = await session.run(
                    query, repository_id=self.repository_id, node_id=node_id
                )
                single_result = await result.single()
                return single_result["node_exists"]
        except Exception as e:
            logger.error(f"Error in has_node: {str(e)}")
            raise

    @db_retry_decorator()
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        try:
            source_node_id = source_node_id.strip('"')
            target_node_id = target_node_id.strip('"')
            query = f"""
                MATCH (a:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
                    -[r {{repository_id: $repository_id}}]->
                    (b:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
                RETURN COUNT(r) > 0 AS edgeExists
            """
            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    repository_id=self.repository_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
                single_result = await result.single()
                return single_result["edgeExists"]
        except Exception as e:
            logger.error(f"Error in has_edge: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_node(self, node_id: str) -> Union[dict, None]:
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            RETURN n
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    query, repository_id=self.repository_id, node_id=node_id
                )
                record = await result.single()
                if record:
                    node = record["n"]
                    node_dict = dict(node)
                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                    )
                    return node_dict
                return None
        except Exception as e:
            logger.error(f"Error in get_node: {str(e)}")
            raise

    @db_retry_decorator()
    async def node_degree(self, node_id: str) -> int:
        node_id = node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
            RETURN COUNT{{ (n)--() }} AS totalEdgeCount
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    query, repository_id=self.repository_id, node_id=node_id
                )
                record = await result.single()
                if record:
                    edge_count = record["totalEdgeCount"]
                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                    )
                    return edge_count
                else:
                    return 0
        except Exception as e:
            logger.error(f"Error in node_degree: {str(e)}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_id = src_id.strip('"')
        tgt_id = tgt_id.strip('"')
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    @db_retry_decorator()
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """
        Find a specific edge between two nodes identified by their node_ids within the same repository.

        Args:
            source_node_id (str): node_id of the source node
            target_node_id (str): node_id of the target node

        Returns:
            dict: Properties of the found relationship, or None if not found
        """
        source_node_id = source_node_id.strip('"')
        target_node_id = target_node_id.strip('"')
        query = f"""
            MATCH (start:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
                -[r {{repository_id: $repository_id}}]->
                (end:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
            RETURN properties(r) AS edge_properties
            LIMIT 1
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    query,
                    repository_id=self.repository_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
                record = await result.single()
                if record:
                    edge_properties = dict(record["edge_properties"])
                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_properties}"
                    )
                    return edge_properties
                else:
                    return None
        except Exception as e:
            logger.error(f"Error in get_edge: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        """
<<<<<<< HEAD
        Retrieves all edges (relationships) for a particular node identified by its node_id within the same repository.
=======
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
        async with self._driver.session() as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database.
>>>>>>> upstream/main

        Args:
            source_node_id (str): node_id of the source node

        Returns:
            List of tuples containing source node_id and connected node_ids
        """
        source_node_id = source_node_id.strip('"')
        query = f"""
            MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
            OPTIONAL MATCH (n)-[r]->(connected:{self.node_label} {{repository_id: $repository_id}})
            RETURN n.node_id AS source_node_id, connected.node_id AS connected_node_id
        """
        try:
            async with self._driver.session() as session:
                results = await session.run(
                    query,
                    repository_id=self.repository_id,
                    source_node_id=source_node_id,
                )
                edges = []
                async for record in results:
                    source_node_id = record["source_node_id"]
                    connected_node_id = record["connected_node_id"]
                    if source_node_id and connected_node_id:
                        edges.append((source_node_id, connected_node_id))

                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edges}"
                )
                return edges
        except Exception as e:
            logger.error(f"Error in get_node_edges: {str(e)}")
            raise

    @db_retry_decorator()
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database with repository isolation.
        """
        try:
            node_id = node_id.strip('"')
            properties = {
                **node_data,
                "repository_id": self.repository_id,
                "node_id": node_id,
            }

            async def _do_upsert(tx: AsyncManagedTransaction):
                query = f"""
                MERGE (n:{self.node_label} {{node_id: $node_id, repository_id: $repository_id}})
                SET n += $properties
                """
                await tx.run(
                    query,
                    node_id=node_id,
                    repository_id=self.repository_id,
                    properties=properties,
                )

            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @db_retry_decorator()
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their node_ids within the same repository.

        Args:
            source_node_id (str): node_id of the source node
            target_node_id (str): node_id of the target node
            edge_data (dict): Dictionary of properties to set on the edge
        """
        try:
            relationship_type = "DIRECTED"
            source_node_id = source_node_id.strip('"')
            target_node_id = target_node_id.strip('"')
            edge_properties = {**edge_data, "repository_id": self.repository_id}

            async def _do_upsert_edge(tx: AsyncManagedTransaction):
                query = f"""
                MATCH (source:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
                MATCH (target:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
                MERGE (source)-[r:{relationship_type} {{repository_id: $repository_id}}]->(target)
                SET r += $properties
                RETURN r
                """
                await tx.run(
                    query,
                    repository_id=self.repository_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    properties=edge_properties,
                )

            async with self._driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    @db_retry_decorator()
    async def delete_node(self, node_id: str):
        try:
            node_id = node_id.strip('"')
            async with self._driver.session() as session:
                query = f"""
                    MATCH (n:{self.node_label} {{repository_id: $repository_id, node_id: $node_id}})
                    DETACH DELETE n
                """
                await session.run(
                    query, repository_id=self.repository_id, node_id=node_id
                )
                logger.debug(f"Node {node_id} deleted from the graph.")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @db_retry_decorator()
    async def delete_edge(self, source_node_id: str, target_node_id: str):
        try:
            source_node_id = source_node_id.strip('"')
            target_node_id = target_node_id.strip('"')
            relationship_type = "DIRECTED"
            async with self._driver.session() as session:
                query = f"""
                    MATCH (source:{self.node_label} {{repository_id: $repository_id, node_id: $source_node_id}})
                        -[r:{relationship_type} {{repository_id: $repository_id}}]->
                        (target:{self.node_label} {{repository_id: $repository_id, node_id: $target_node_id}})
                    DELETE r
                """
                await session.run(
                    query,
                    repository_id=self.repository_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
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
        """
        Get all nodes that have a specific property value within the same repository.

        Args:
            property_name (str): The name of the property to match
            property_value (Any): The value to match against
            split_by_sep (bool): If True, treats property value as GRAPH_FIELD_SEP-separated
                                string and matches if value exists in any part
        Returns:
            List[Dict]: List of node dictionaries matching the criteria
        """
        try:
            async with self._driver.session() as session:
                if split_by_sep:
                    query = f"""
                        MATCH (n:{self.node_label} {{repository_id: $repository_id}})
                        WHERE n[$property_name] IS NOT NULL
                        AND ANY(x IN SPLIT(n[$property_name], $sep) WHERE x = $property_value)
                        RETURN n
                    """
                    params = {
                        "repository_id": self.repository_id,
                        "property_name": property_name,
                        "property_value": property_value,
                        "sep": GRAPH_FIELD_SEP,
                    }
                else:
                    query = f"""
                        MATCH (n:{self.node_label} {{repository_id: $repository_id}})
                        WHERE n[$property_name] = $property_value
                        RETURN n
                    """
                    params = {
                        "repository_id": self.repository_id,
                        "property_name": property_name,
                        "property_value": property_value,
                    }

                result = await session.run(query, **params)
                nodes = []
                async for record in result:
                    node = dict(record["n"])
                    nodes.append(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{nodes}"
                )
                return nodes
        except Exception as e:
            logger.error(f"Error in get_nodes_by_property: {str(e)}")
            raise

    @db_retry_decorator()
    async def get_edges_by_property(
        self, property_name: str, property_value: Any, split_by_sep: bool = False
    ) -> List[Dict]:
        """
        Get all edges that have a specific property value within the same repository.

        Args:
            property_name (str): The name of the property to match
            property_value (Any): The value to match against
            split_by_sep (bool): If True, treats property value as GRAPH_FIELD_SEP-separated
                                string and matches if value exists in any part
        Returns:
            List[Dict]: List of edge dictionaries matching the criteria
        """
        try:
            async with self._driver.session() as session:
                if split_by_sep:
                    # Use SPLIT and ANY for GRAPH_FIELD_SEP separated values
                    query = """
                        MATCH ()-[r]->()
                        WHERE r.repository_id = $repository_id
                        AND r[$property_name] IS NOT NULL
                        AND ANY(x IN SPLIT(r[$property_name], $sep) WHERE x = $property_value)
                        RETURN r
                    """
                    params = {
                        "repository_id": self.repository_id,
                        "property_name": property_name,
                        "property_value": property_value,
                        "sep": GRAPH_FIELD_SEP,
                    }
                else:
                    # Use exact matching
                    query = """
                        MATCH ()-[r:RELATIONSHIP_TYPE]->()
                        WHERE r.repository_id = $repository_id
                        AND r[$property_name] = $property_value
                        RETURN r
                    """
                    params = {
                        "repository_id": self.repository_id,
                        "property_name": property_name,
                        "property_value": property_value,
                    }

                result = await session.run(query, **params)
                edges = []
                async for record in result:
                    edge = dict(record["r"])
                    edges.append(edge)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edges}"
                )
                return edges
        except Exception as e:
            logger.error(f"Error in get_edges_by_property: {str(e)}")
            raise

    @db_retry_decorator()
    async def drop(self):
        """
        Deletes all nodes and edges associated with the current repository_id
        and closes the database connection.
        """
        try:
            # Step 1: Delete all nodes and relationships for the current repository_id
            async with self._driver.session() as session:
                query = f"""
                    MATCH (n:{self.node_label} {{repository_id: $repository_id}})
                    DETACH DELETE n
                """
                await session.run(query, repository_id=self.repository_id)
                logger.info(
                    f"All nodes and edges for repository_id '{self.repository_id}' have been deleted."
                )

            # Step 2: Close the connection
            await self.close()
            logger.info(
                f"Neo4j connection closed after cleaning up repository_id '{self.repository_id}'."
            )
        except Exception as e:
            logger.error(
                f"Error during drop operation for repository_id '{self.repository_id}': {str(e)}"
            )
            raise
