from dataclasses import dataclass
import os
import hashlib
import atexit

from qdrant_client import QdrantClient
from qdrant_client.http import models
from grpc import StatusCode
from grpc._channel import _InactiveRpcError

from ..base import BaseVectorStorage
from ..utils import compute_mdhash_id, logger
from tqdm.asyncio import tqdm as tqdm_async
import asyncio
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class QdrantStorage(BaseVectorStorage):
    """Qdrant vector storage implementation with multi-tenancy support."""

    cosine_better_than_threshold: float = 0.3

    db_retry = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (_InactiveRpcError, ConnectionError, TimeoutError)
        ),
    )

    def __post_init__(self):
        self.environment = self.global_config.get("environment", "dev")
        self._collection_name = f"lightrag_{self.namespace}_{self.environment}"
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.repository = self.global_config.get("repository_name")
        self.repository_id = self.global_config.get("repository_id")

        if not self.repository or not self.repository_id:
            raise ValueError(
                "repository and repository_id are required in global_config"
            )

        # Initialize Qdrant client
        try:
            self._client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=True,
            )
            # Create collection if it doesn't exist
            try:
                self._client.get_collection(self._collection_name)
                logger.info(
                    f"Connected to existing Qdrant collection {self._collection_name}"
                )
            except _InactiveRpcError as e:
                if e.code() == StatusCode.NOT_FOUND:
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=models.VectorParams(
                            size=self.embedding_func.embedding_dim,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name="repository_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    logger.info(
                        f"Created new Qdrant collection {self._collection_name}"
                    )
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

        # Register cleanup method to run at shutdown
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Cleanup method to properly close the gRPC channel"""
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

    def _get_qdrant_id(self, string_id: str) -> int:
        """Convert string ID to integer for Qdrant.
        Creates a deterministic integer based on the string ID and repository_id.
        """
        # Create a deterministic hash combining repository_id and string_id
        hash_input = f"{self.repository_id}:{string_id}"
        # Use last 16 digits of hash for the integer (to avoid overflow)
        hash_hex = hashlib.md5(hash_input.encode()).hexdigest()[-16:]
        return int(hash_hex, 16)

    @db_retry
    async def upsert(self, data: dict[str, dict]):
        """Insert or update vectors in Qdrant with repository_id."""
        logger.info(
            f"Inserting {len(data)} vectors to {self.namespace} for repository {self.repository_id}"
        )
        if not data:
            logger.warning("Attempting to insert empty data to vector DB")
            return []

        # Prepare all data upfront
        list_data = []
        contents = []
        for doc_id, doc in data.items():
            payload = {k: v for k, v in doc.items() if k in self.meta_fields}
            payload["repository_id"] = self.repository_id
            list_data.append((doc_id, payload))
            contents.append(doc["content"])

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)

        points = [
            models.PointStruct(
                id=self._get_qdrant_id(doc_id),
                vector=embeddings[i],
                payload={
                    **payload,
                    "original_id": doc_id,
                },
            )
            for i, (doc_id, payload) in enumerate(list_data)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        return points

    @db_retry
    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        """Query vectors from Qdrant within the same repository."""
        try:
            embedding = await self.embedding_func([query])
            repository_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="repository_id",
                        match=models.MatchValue(value=self.repository_id),
                    )
                ]
            )

            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=embedding[0],
                limit=top_k,
                score_threshold=self.cosine_better_than_threshold,
                query_filter=repository_filter,  # Uncomment and use the filter
            )

            return [
                {
                    **hit.payload,
                    "id": hit.payload["original_id"],
                    "score": round(hit.score, 4),
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error during query operation: {e}")
            raise

    @db_retry
    async def query_by_id(self, id: str) -> dict | None:
        try:
            # Search using payload filter for original_id
            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=self.repository_id),
                        ),
                        models.FieldCondition(
                            key="original_id", match=models.MatchValue(value=id)
                        ),
                    ]
                ),
                limit=1,
            )

            if results[0]:
                hit = results[0][0]
                return {**hit.payload, "id": hit.payload["original_id"]}

            return None
        except Exception as e:
            logger.error(f"Error during query_by_id operation: {e}")
            raise

    @db_retry
    async def delete_by_ids(self, ids: list[str]):
        """Delete vectors by their IDs within the same repository."""
        try:
            # Convert string IDs to Qdrant integer IDs
            qdrant_ids = [self._get_qdrant_id(id) for id in ids]

            # First verify these IDs belong to the current repository
            points = self._client.retrieve(
                collection_name=self._collection_name, ids=qdrant_ids
            )

            # Filter points that belong to this repository
            valid_ids = [
                point.id
                for point in points
                if point.payload.get("repository_id") == self.repository_id
            ]

            if valid_ids:
                self._client.delete(
                    collection_name=self._collection_name, points_selector=valid_ids
                )
                logger.debug(
                    f"Deleted {len(valid_ids)} points from collection {self._collection_name}"
                )
            else:
                logger.warning(
                    f"No valid points found for deletion in repository {self.repository_id}"
                )
        except Exception as e:
            logger.error(f"Error during delete operation: {e}")
            raise

    @db_retry
    async def delete_entity(self, entity_name: str):
        """Delete an entity and its associated vectors within the same repository."""
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            qdrant_id = self._get_qdrant_id(entity_id)

            # Check if entity exists and belongs to current repository
            search_result = self._client.retrieve(
                collection_name=self._collection_name, ids=[qdrant_id]
            )

            if (
                search_result
                and search_result[0].payload.get("repository_id") == self.repository_id
            ):
                await self.delete_by_ids([entity_id])
                logger.info(
                    f"Entity {entity_name} has been deleted from repository {self.repository_id}"
                )
            else:
                logger.info(
                    f"No entity found with name {entity_name} in repository {self.repository_id}"
                )
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    @db_retry
    async def delete_relation(self, entity_name: str):
        """Delete all relations associated with an entity within the same repository."""
        try:
            # Search for relations containing the entity within the same repository
            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=self.repository_id),
                        )
                    ],
                    should=[
                        models.FieldCondition(
                            key="src_id", match=models.MatchValue(value=entity_name)
                        ),
                        models.FieldCondition(
                            key="tgt_id", match=models.MatchValue(value=entity_name)
                        ),
                    ],
                ),
            )

            if results[0]:  # If there are any results
                ids_to_delete = [point.id for point in results[0]]
                await self.delete_by_ids(ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} relations related to entity {entity_name} in repository {self.repository_id}"
                )
            else:
                logger.info(
                    f"No relations found for entity {entity_name} in repository {self.repository_id}"
                )
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        """Called when indexing is complete."""
        # Qdrant handles persistence automatically
        pass

    @db_retry
    async def drop(self):
        """Delete all the points in this repository."""
        try:
            # Delete points only for the current repository
            repository_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="repository_id",
                        match=models.MatchValue(value=self.repository_id),
                    )
                ]
            )

            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=repository_filter),
            )
            logger.info(
                f"Deleted all points for repository {self.repository_id} in collection {self._collection_name}"
            )
        except Exception as e:
            logger.error(f"Error while dropping repository data: {e}")
            raise
