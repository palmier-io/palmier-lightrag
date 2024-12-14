from ..base import BaseKVStorage
from ..utils import logger
from typing import Optional
from os import getenv
from supabase import create_client
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from postgrest.exceptions import APIError
from httpx import HTTPError, TimeoutException


@dataclass
class SupabaseChunksStorage(BaseKVStorage):
    db_retry = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((APIError, HTTPError, TimeoutException)),
    )

    def __post_init__(self):
        # Get Supabase credentials from environment variables
        supabase_url = getenv("SUPABASE_URL")
        supabase_key = getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )

        environment = self.global_config.get("environment", "dev")

        # Get storage params, raising error if not present
        storage_params = self.global_config.get("storage_params")
        if not storage_params:
            raise ValueError("storage_params must be provided in global_config")

        supabase_params = storage_params.get("supabase")
        if not supabase_params:
            raise ValueError(
                "supabase configuration must be provided in storage_params"
            )

        # Get required parameters, raising specific errors if any are missing
        try:
            self.table_name = f"{supabase_params['table_name']}_{environment}"
            self.repo = storage_params["repository"]
            self.repo_id = storage_params["repository_id"]
        except KeyError as e:
            raise ValueError(
                f"Missing required parameter in supabase config: {e.args[0]}"
            )

        # Create Supabase client
        self.client = create_client(supabase_url, supabase_key)
        self.table = self.client.table(self.table_name)
        logger.info(f"Initialized Supabase chunks storage for repository {self.repo}")

    @db_retry
    async def all_keys(self) -> list[str]:
        """List all keys (naturally idempotent - read-only operation)"""
        try:
            response = (
                self.table.select("chunk_id").eq("repository", self.repo).execute()
            )
            return [row["chunk_id"] for row in response.data]
        except Exception as e:
            logger.error(f"Error in all_keys: {str(e)}")
            raise

    @db_retry
    async def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Get by ID (naturally idempotent - read-only operation)"""
        try:
            response = (
                self.table.select("*")
                .eq("repository_id", self.repo_id)
                .eq("chunk_id", chunk_id)
                .execute()
            )
            if not response.data:
                return None

            row = response.data[0]
            return {
                "content": row["content"],
                "file_path": row["file_path"],
                "full_doc_id": row["full_doc_id"],
                **(row["metadata"] or {}),
            }
        except Exception as e:
            logger.error(f"Error in get_by_id: {str(e)}")
            raise

    async def get_by_ids(
        self, chunk_ids: list[str], fields: Optional[set[str]] = None
    ) -> list[Optional[dict]]:
        response = (
            self.table.select("*")
            .eq("repository_id", self.repo_id)
            .in_("chunk_id", chunk_ids)
            .execute()
        )

        # Create mapping of chunk_id to data
        id_to_data = {}
        for row in response.data:
            data = {
                "content": row["content"],
                "file_path": row["file_path"],
                "full_doc_id": row["full_doc_id"],
                **(row["metadata"] or {}),
            }
            if fields:
                data = {k: v for k, v in data.items() if k in fields}
            id_to_data[row["chunk_id"]] = data

        return [id_to_data.get(chunk_id) for chunk_id in chunk_ids]

    async def get_by_field(self, field: str, values: list) -> dict[str, dict]:
        """Optimized for doc_id queries"""
        if field == "full_doc_id":
            # Use the optimized index
            response = (
                self.table.select("*")
                .eq("repository_id", self.repo_id)
                .in_("full_doc_id", values)
                .execute()
            )
        else:
            # Fallback to filtering metadata for other fields
            response = (
                self.table.select("*").eq("repository_id", self.repo_id).execute()
            )

        result = {}
        for row in response.data:
            # Check if the field value matches (either in main columns or metadata)
            field_value = row.get(field) or (row["metadata"] or {}).get(field)
            if field_value in values:
                result[row["chunk_id"]] = {
                    "content": row["content"],
                    "file_path": row["file_path"],
                    "full_doc_id": row["full_doc_id"],
                    **(row["metadata"] or {}),
                }
        return result

    async def filter_keys(self, chunk_ids: list[str]) -> set[str]:
        """Return non-existent chunk_ids"""
        existing = (
            self.table.select("chunk_id")
            .eq("repository_id", self.repo_id)
            .in_("chunk_id", chunk_ids)
            .execute()
        )
        existing_ids = {row["chunk_id"] for row in existing.data}
        return set(chunk_ids) - existing_ids

    @db_retry
    async def upsert(self, data: dict[str, dict]) -> dict[str, dict]:
        """
        Insert or update chunks (idempotent using ON CONFLICT DO UPDATE)
        """
        try:
            # Convert to database format
            supabase_data = []
            for chunk_id, chunk_data in data.items():
                # Extract core fields
                db_row = {
                    "repository_id": self.repo_id,
                    "full_doc_id": chunk_data["full_doc_id"],
                    "chunk_id": chunk_id,
                    "repository": self.repo,
                    "file_path": chunk_data["file_path"],
                    "content": chunk_data["content"],
                }

                # Put remaining fields in metadata
                metadata = {
                    k: v
                    for k, v in chunk_data.items()
                    if k not in {"full_doc_id", "content", "file_path"}
                }
                if metadata:
                    db_row["metadata"] = metadata

                supabase_data.append(db_row)

            self.table.upsert(supabase_data).execute()
            return data
        except Exception as e:
            logger.error(f"Error in upsert: {str(e)}")
            raise

    async def drop(self):
        self.table.delete().eq("repository_id", self.repo_id).execute()

    async def delete_by_ids(self, chunk_ids: list[str]):
        self.table.delete().eq("repository_id", self.repo_id).in_(
            "chunk_id", chunk_ids
        ).execute()

    async def index_done_callback(self):
        # No-op as data is already persisted
        pass


"""
Supabase storage implementation for LightRAG.

Schema:
```
CREATE TABLE lightrag_text_chunks (
    repository_id text NOT NULL,
    full_doc_id text NOT NULL,
    chunk_id text NOT NULL,
    repository text NOT NULL,
    file_path text NOT NULL,
    content text NOT NULL,
    metadata jsonb,
    PRIMARY KEY (repository_id, chunk_id),
    CONSTRAINT idx_chunks_doc_id UNIQUE (repository_id, full_doc_id, chunk_id)
);
```

The table uses repository for partitioning data from different repositories,
with chunk_id as the primary identifier and doc_id for grouping chunks.
"""
