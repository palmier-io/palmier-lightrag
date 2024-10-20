from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import os
import asyncio

WORKING_DIR = "./ragtest"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

async def naive_rag_search(query: str) -> str:
    """
    Perform a naive RAG search with the given query and return the response.
    
    Args:
        query (str): The query string to search for.
    
    Returns:
        str: The response from the RAG search.
    """
    return await rag.aquery(query, param=QueryParam(mode="naive"))
