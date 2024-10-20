import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class SchemaProperty(BaseModel):
    type: str
    description: str
    x_explanation: str = Field(alias="x-explanation")

class Schema(BaseModel):
    type: str = "object"
    properties: Dict[str, SchemaProperty]
    required: list[str] = []

class SchemaManager:
    def __init__(self, schema_folder: str = "api/schemas"):
        self.schema_folder = schema_folder
        os.makedirs(self.schema_folder, exist_ok=True)

    def create_schema(self, name: str, schema: Dict[str, Any], description: str) -> bool:
        file_path = os.path.join(self.schema_folder, f"{name}.json")
        print(file_path)
        if os.path.exists(file_path):
            return False
        new_schema = Schema(
            type="object",
            properties={
                k: SchemaProperty(**v) for k, v in schema.get("properties", {}).items()
            },
            required=schema.get("required", [])
        )
        schema_data = {
            "schema": new_schema.model_dump(),
            "description": description
        }
        with open(file_path, "w") as f:
            json.dump(schema_data, f, indent=2)
        return True

    def get_schema_by_name(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        if name is None:
            name = "default"
        file_path = os.path.join(self.schema_folder, f"{name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    def update_schema(self, name: str, schema: Dict[str, Any], description: str) -> bool:
        file_path = os.path.join(self.schema_folder, f"{name}.json")
        if not os.path.exists(file_path):
            return False
        updated_schema = Schema(
            type="object",
            properties={
                k: SchemaProperty(**v) for k, v in schema.get("properties", {}).items()
            },
            required=schema.get("required", [])
        )
        schema_data = {
            "schema": updated_schema.dict(),
            "description": description
        }
        with open(file_path, "w") as f:
            json.dump(schema_data, f, indent=2)
        return True

    def delete_schema(self, name: str) -> bool:
        file_path = os.path.join(self.schema_folder, f"{name}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def list_schemas(self) -> Dict[str, str]:
        schemas = {}
        for filename in os.listdir(self.schema_folder):
            if filename.endswith(".json"):
                name = filename[:-5]  # Remove .json extension
                file_path = os.path.join(self.schema_folder, filename)
                with open(file_path, "r") as f:
                    schema_data = json.load(f)
                schemas[name] = schema_data["description"]
        return schemas

# Example usage:
def create_example_schemas():
    schema_manager = SchemaManager()

    # Company Profile Schema
    company_profile_schema = {
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the company",
                "x-explanation": "Provide the full legal name of the company"
            },
            "industry": {
                "type": "string",
                "description": "The industry the company operates in",
                "x-explanation": "Specify the primary industry sector of the company"
            },
            "employees": {
                "type": "integer",
                "description": "The number of employees",
                "x-explanation": "Provide the most recent count of full-time employees"
            },
            "revenue": {
                "type": "number",
                "description": "Annual revenue in millions USD",
                "x-explanation": "State the company's annual revenue in millions of US dollars"
            }
        },
        "required": ["name", "industry"]
    }
    schema_manager.create_schema("company_profile", company_profile_schema, "Company Profile Schema")

    # Market Trends Schema
    market_trends_schema = {
        "properties": {
            "trend_name": {
                "type": "string",
                "description": "Name of the market trend",
                "x-explanation": "Provide a concise name for the identified market trend"
            },
            "description": {
                "type": "string",
                "description": "Description of the trend",
                "x-explanation": "Explain the trend, its causes, and potential impacts"
            },
            "impact_score": {
                "type": "integer",
                "description": "Impact score from 1-10",
                "x-explanation": "Rate the potential impact of this trend on a scale of 1 (low) to 10 (high)"
            }
        },
        "required": ["trend_name", "description"]
    }
    schema_manager.create_schema("market_trends", market_trends_schema, "Market Trends Schema")

    return schema_manager

# Uncomment to create example schemas
# example_schema_manager = create_example_schemas()
