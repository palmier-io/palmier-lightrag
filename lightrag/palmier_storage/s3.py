import boto3
import json
import os
from dataclasses import dataclass
from typing import Union, Any
from ..base import BaseKVStorage
from ..utils import logger

@dataclass
class S3DocsStorage(BaseKVStorage):
    def __post_init__(self):

        aws_access_key_id = os.getenv("AWS_S3_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_S3_SECRET_ACCESS_KEY")
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS_S3_ACCESS_KEY_ID and AWS_S3_SECRET_ACCESS_KEY must be set in environment variables")

        # Get storage params, raising error if not present
        storage_params = self.global_config.get("storage_params")
        if not storage_params:
            raise ValueError("storage_params must be provided in global_config")
            
        s3_params = storage_params.get("s3")
        if not s3_params:
            raise ValueError("s3 configuration must be provided in storage_params")
        
        # Get required parameters
        try:
            self.bucket_name = s3_params["bucket_name"]
            self.repository = storage_params["repository"]
            self.repository_id = storage_params["repository_id"]

            # Add any other required S3 parameters here, for example:
            # self.region = s3_params["region"]
            # self.endpoint_url = s3_params["endpoint_url"]
        except KeyError as e:
            raise ValueError(f"Missing required parameter in s3 config: {e.args[0]}")

        self.prefix = f"{self.repository}/"
        
        # Initialize S3 client with parameters
        try:
            self.s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            # Ensure bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except Exception as e:
            logger.error(f"Error accessing S3 bucket {self.bucket_name}: {e}")
            raise

    async def all_keys(self) -> list[str]:
        """List all keys in the storage"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            keys = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' in page:
                    keys.extend(obj['Key'].replace(self.prefix, '') for obj in page['Contents'])
            return keys
        except Exception as e:
            logger.error(f"Error listing keys from S3: {e}")
            return []

    async def get_by_id(self, id: str) -> Union[Any, None]:
        """Get a single item by ID"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=f"{self.prefix}{id}"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Error getting item {id} from S3: {e}")
            return None

    async def get_by_ids(self, ids: list[str], fields: Union[set[str], None] = None) -> list[Union[Any, None]]:
        """Get multiple items by IDs"""
        results = []
        for id in ids:
            item = await self.get_by_id(id)
            if item and fields:
                item = {k: v for k, v in item.items() if k in fields}
            results.append(item)
        return results

    async def get_by_field(self, field: str, values: list[str]) -> dict[str, dict]:
        """Get items by field value"""
        # Note: This is inefficient for S3 as we need to scan all objects
        # Consider using a secondary index or different storage for this use case
        all_keys = await self.all_keys()
        result = {}
        values_set = set(values)
        
        for key in all_keys:
            item = await self.get_by_id(key)
            if item and item.get(field) in values_set:
                result[key] = item
                
        return result

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return keys that don't exist in storage"""
        existing_keys = set(await self.all_keys())
        return set(data) - existing_keys

    async def upsert(self, data: dict[str, dict]):
        """Insert or update items"""
        for key, value in data.items():
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{self.prefix}{key}",
                    Body=json.dumps(value).encode('utf-8'),
                    ContentType='application/json'
                )
            except Exception as e:
                logger.error(f"Error upserting item {key} to S3: {e}")
        return data

    async def drop(self):
        """Delete all items in the namespace"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects}
                    )
        except Exception as e:
            logger.error(f"Error dropping S3 storage: {e}")

    async def delete_by_ids(self, ids: list[str]):
        """Delete items by IDs"""
        try:
            objects = [{'Key': f"{self.prefix}{id}"} for id in ids]
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects}
            )
        except Exception as e:
            logger.error(f"Error deleting items from S3: {e}")

    async def index_done_callback(self):
        """No-op for S3 as changes are immediate"""
        pass
