from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import os
import sys

WORKING_DIR = "./ragtest"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

if len(sys.argv) < 2:
    query = input("Please enter your query: ")
    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)
else:
    query = sys.argv[1]

# Perform naive search
print("Performing naive search...\n")
print(rag.query(query, param=QueryParam(mode="naive")))
# Perform local search
print("Performing local search...\n")
print(rag.query(query, param=QueryParam(mode="local")))
# Perform global search
print("Performing global search...\n")
print(rag.query(query, param=QueryParam(mode="global")))
# Perform hybrid search
print("Performing hybrid search...\n")
print(rag.query(query, param=QueryParam(mode="hybrid")))
