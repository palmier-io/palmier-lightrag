from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from query import naive_rag_search
from schema_manager import SchemaManager
from llm_formatter import format_response_with_llm

app = FastAPI()
schema_manager = SchemaManager()

class Schema(BaseModel):
    schema: Dict[str, Any]
    name: str
    description: str

class QueryRequest(BaseModel):
    query: str
    schema_name: Optional[str] = None
    custom_schema: Optional[Dict[str, Any]] = None
    additional_params: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/query/")
async def query(request: QueryRequest):
    # Use the fields from the request model
    raw_results = await naive_rag_search(request.query)
    # Get schema
    schema = None
    if request.schema_name:
        schema = schema_manager.get_schema_by_name(request.schema_name)
    elif request.custom_schema:
        schema = request.custom_schema
    
    # Format response using LLM, including the query
    formatted_response = await format_response_with_llm(raw_results, schema, request.additional_params, query=request.query)
    
    return {"result": formatted_response}

@app.post("/schema/")
async def create_schema(schema: Schema):
    schema_id = schema_manager.create_schema(schema.name, schema.schema, schema.description)
    return {"schema_id": schema_id}

@app.get("/schema/{schema_name}")
async def get_schema(schema_name: str):
    schema = schema_manager.get_schema_by_name(schema_name)
    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")
    return schema

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug"
    )
