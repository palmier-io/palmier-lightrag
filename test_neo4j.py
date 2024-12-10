import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.chunking.code_chunker import traverse_directory

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./local_palmier"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    graph_storage="Neo4JStorage",
    log_level="INFO",
    chunk_summary_enabled=True,
    # vector_storage="QdrantStorage",
    # docs_storage="S3DocsStorage",
    storage_params={
        "repository": "palmier-io/palmier",
        # "repository_id": 882116377,
        "repository_id": 883418417,
        "s3": {"bucket_name": "lightrag-docs"},
        "supabase": {
            "table_name": "lightrag_text_chunks",
        },
    },
    environment="dev",
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)


repo_path = "/Users/harrison/palmier-io/palmier"
file_paths = traverse_directory(repo_path)
rag.insert_files(directory=repo_path, file_paths=file_paths)

# with open("./book.txt") as f:
#     rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# Perform hybrid search
print(
    rag.query(
        "What is palmier?",
        param=QueryParam(mode="hybrid", top_k=10, only_need_context=True),
    )
)

# rag.delete_files(
#     directory="/Users/harrison/palmier-io/palmier-lightrag", file_paths=["book.txt"]
# )

# rag.drop()
