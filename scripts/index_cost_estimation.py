import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tempfile
import subprocess
from lightrag.chunking.code_chunker import CodeChunker
from lightrag.utils import logger
from lightrag.prompt import PROMPTS
from lightrag.utils import encode_string_by_tiktoken
from lightrag import LightRAG
from lightrag.llm import azure_openai_embedding, azure_openai_complete
from tqdm import tqdm
from openai.types.completion_usage import CompletionUsage

from dotenv import load_dotenv

load_dotenv()

def clone_repo(repo_url: str, temp_dir: str, commit_hash: str = None) -> str:
    """Clone a GitHub repository to a temporary directory.
    
    Args:
        repo_url: Repository in format 'owner/repo' or full URL
        temp_dir: Directory to clone into
        commit_hash: Optional specific commit to checkout
    """
    logger.info(f"Cloning {repo_url} to {temp_dir}")
    
    # Convert owner/repo format to full URL
    if '/' in repo_url and not repo_url.startswith('http'):
        repo_url = f"https://github.com/{repo_url}"
    
    # For HTTPS authentication
    if repo_url.startswith("https://"):
        token = os.getenv("GITHUB_TOKEN")
        if token:
            repo_url = repo_url.replace("https://", f"https://{token}@")
    
    # Clone the repository
    subprocess.run(['git', 'clone', repo_url, temp_dir], check=True)
    
    # Checkout specific commit if provided
    if commit_hash:
        logger.info(f"Checking out commit: {commit_hash}")
        subprocess.run(['git', 'checkout', commit_hash], cwd=temp_dir, check=True)
    
    return temp_dir

def get_all_files(directory: str) -> list[str]:
    """Get all files in directory recursively."""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# TODO: read through and ensure that it matches with extract_entities in operate.py
def estimate_entity_extraction_cost(chunks: list[dict], model_name: str = "gpt-4o-mini") -> dict:
    """Estimate the cost of running entity extraction and merging on chunks."""
    MODEL_COSTS = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    }
    
    if model_name not in MODEL_COSTS:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get token counts for prompts
    base_prompt = PROMPTS["entity_extraction"].format(
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        input_text=""  # Will add content length separately
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    
    base_prompt_tokens = len(encode_string_by_tiktoken(base_prompt))
    continue_prompt_tokens = len(encode_string_by_tiktoken(continue_prompt))
    if_loop_prompt_tokens = len(encode_string_by_tiktoken(if_loop_prompt))
    
    # Initialize counters
    total_input_tokens = 0
    total_output_tokens = 0
    total_api_calls = 0
    
    # For each chunk, we make:
    # 1. Initial extraction call
    # 2. Up to entity_extract_max_gleaning continue calls
    # 3. Up to entity_extract_max_gleaning if_loop calls
    MAX_GLEANING = 1  # Typical value for entity_extract_max_gleaning
    MERGE_ENTITIES = False
    
    for chunk in chunks:
        # Initial extraction call
        content_tokens = len(encode_string_by_tiktoken(chunk["content"]))
        total_input_tokens += content_tokens + base_prompt_tokens
        total_output_tokens += 700  # Estimate for entity/relation extraction TODO: find better estimate
        total_api_calls += 1

        previous_context_tokens = content_tokens + base_prompt_tokens + 700
        
        # Estimate gleaning calls
        for _ in range(MAX_GLEANING):
            # Continue prompt with previous context
            total_input_tokens += continue_prompt_tokens + previous_context_tokens  # Previous context estimate
            total_output_tokens += 650  # Additional entities/relations TODO: find better estimate
            total_api_calls += 1
            previous_context_tokens += continue_prompt_tokens + 500 # Additional entities/relations
            
            # If-loop prompt with previous context
            '''
            total_input_tokens += if_loop_prompt_tokens + previous_context_tokens 
            total_output_tokens += 10  # Yes/No response
            total_api_calls += 1
            '''
    
    if MERGE_ENTITIES:
        # Get token count for summary prompt
        summary_prompt = PROMPTS["summarize_entity_descriptions"].format(
            entity_name="EXAMPLE_ENTITY",
            description_list=[""]
        )
        summary_prompt_tokens = len(encode_string_by_tiktoken(summary_prompt))
        
        # Initialize counters for merging costs
        merge_input_tokens = 0
        merge_output_tokens = 0
        merge_api_calls = 0
        
        # Estimate entity merging costs
        estimated_descriptions_per_entity = 2 # TODO: find better estimate
        estimated_average_description_length = 2000 # TODO: find better estimate
        estimated_unique_entities = len(chunks) * 2 # TODO: find better estimate
        
        # For each entity needing summary:
        # 1. Input includes summary prompt + combined descriptions
        # 2. Output is the summarized description (limited by entity_summary_to_max_tokens)
        SUMMARY_MAX_TOKENS = 500  # Typical value for entity_summary_to_max_tokens
        merge_input_tokens += (summary_prompt_tokens + (estimated_average_description_length * estimated_descriptions_per_entity)) * estimated_unique_entities  # Prompt + average combined description length
        merge_output_tokens += SUMMARY_MAX_TOKENS * estimated_unique_entities  # Maximum summary length
        merge_api_calls += estimated_unique_entities
        
        # Estimate relationship merging costs similarly
        estimated_descriptions_per_relationship = 3 # TODO: find better estimate
        estimated_average_relationship_length = 1500 # TODO: find better estimate
        estimated_unique_relationships = len(chunks) * 2 # TODO: find better estimate
        
        merge_input_tokens += (summary_prompt_tokens + (estimated_average_relationship_length * estimated_descriptions_per_relationship)) * estimated_unique_relationships  # Prompt + average combined description length
        merge_output_tokens += SUMMARY_MAX_TOKENS * estimated_unique_relationships  # Maximum summary length
        merge_api_calls += estimated_unique_relationships
        
        # Add merging costs to total
        total_input_tokens += merge_input_tokens
        total_output_tokens += merge_output_tokens
        #total_api_calls += merge_api_calls
    
    # Calculate costs
    input_cost = (total_input_tokens / 1000000) * MODEL_COSTS[model_name]["input"]
    output_cost = (total_output_tokens / 1000000) * MODEL_COSTS[model_name]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model_name,
        "total_chunks": len(chunks),
        "total_api_calls": total_api_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_input_cost": round(input_cost, 2),
        "estimated_output_cost": round(output_cost, 2),
        "estimated_total_cost": round(total_cost, 2),
    }

def calculate_actual_costs(chunks: list[dict], temp_dir: str) -> dict:
    """Calculate actual costs of processing chunks through LightRAG."""
    actual_usage = {
        'insertion': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'api_calls': 0
        }
    }

    try:
        async def track_usage_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            response, usage = await azure_openai_complete(prompt, system_prompt, history_messages, **kwargs)
            print("USAGE: ", usage)
            if isinstance(usage, CompletionUsage):
                actual_usage['insertion']['prompt_tokens'] += usage.prompt_tokens
                actual_usage['insertion']['completion_tokens'] += usage.completion_tokens
                actual_usage['insertion']['total_tokens'] += usage.total_tokens
                actual_usage['insertion']['api_calls'] += 1
                return response
            else:
                logger.error(f"Unexpected usage type: {type(usage)}")
                return response

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=track_usage_complete,
            embedding_func=azure_openai_embedding,
        )

        # Insert all chunks
        for chunk in tqdm(chunks):
            rag.insert(chunk['content'])

        # Calculate costs
        MODEL_COSTS = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        }
        actual_input_cost = (actual_usage['insertion']['prompt_tokens'] / 1000000) * MODEL_COSTS["gpt-4o-mini"]["input"]
        actual_output_cost = (actual_usage['insertion']['completion_tokens'] / 1000000) * MODEL_COSTS["gpt-4o-mini"]["output"]
        actual_total_cost = actual_input_cost + actual_output_cost

        return {
            'api_calls': actual_usage['insertion']['api_calls'],
            'input_tokens': actual_usage['insertion']['prompt_tokens'],
            'output_tokens': actual_usage['insertion']['completion_tokens'],
            'total_tokens': actual_usage['insertion']['total_tokens'],
            'input_cost': round(actual_input_cost, 2),
            'output_cost': round(actual_output_cost, 2),
            'total_cost': round(actual_total_cost, 2)
        }

    except Exception as e:
        logger.error(f"Failed to track actual insertion costs: {e}")
        return None

def estimate_chunking_cost(repo_url: str, commit_hash: str = None):
    """Estimate the cost of chunking and processing a repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone repo and prepare chunks
        repo_dir = clone_repo(repo_url, temp_dir, commit_hash)
        chunker = CodeChunker(
            repo_dir,
            target_tokens=800,
            overlap_token_size=100,
            tiktoken_model="gpt-4o-mini"
        )
        
        file_paths = get_all_files(repo_dir)
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")
        print("NUMBER OF CHUNKS: ", len(all_chunks))
        
        # Get estimates
        cost_estimate = estimate_entity_extraction_cost(all_chunks)

        # Add user confirmation
        response = input("Calculate actual costs? (Y/n): ")
        if response == 'Y':
            # Get actual costs
            actual_costs = calculate_actual_costs(all_chunks, temp_dir)
            if actual_costs:
                cost_estimate['actual_insertion_costs'] = actual_costs
        
        # Log results
        logger.info("Cost Estimation:")
        logger.info(f"Model: {cost_estimate['model']}")
        logger.info(f"Total API calls: {cost_estimate['total_api_calls']:,}")
        logger.info(f"Estimated input tokens: {cost_estimate['estimated_input_tokens']:,}")
        logger.info(f"Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
        logger.info(f"Estimated input cost: ${cost_estimate['estimated_input_cost']:.2f}")
        logger.info(f"Estimated output cost: ${cost_estimate['estimated_output_cost']:.2f}")
        logger.info(f"Estimated total cost: ${cost_estimate['estimated_total_cost']:.2f}")
        
        if 'actual_insertion_costs' in cost_estimate:
            logger.info("\nActual Insertion Costs:")
            logger.info(f"API calls: {cost_estimate['actual_insertion_costs']['api_calls']:,}")
            logger.info(f"Input tokens: {cost_estimate['actual_insertion_costs']['input_tokens']:,}")
            logger.info(f"Output tokens: {cost_estimate['actual_insertion_costs']['output_tokens']:,}")
            logger.info(f"Total tokens: {cost_estimate['actual_insertion_costs']['total_tokens']:,}")
            logger.info(f"Input cost: ${cost_estimate['actual_insertion_costs']['input_cost']:.2f}")
            logger.info(f"Output cost: ${cost_estimate['actual_insertion_costs']['output_cost']:.2f}")
            logger.info(f"Total cost: ${cost_estimate['actual_insertion_costs']['total_cost']:.2f}")
        
        return all_chunks, cost_estimate

if __name__ == "__main__":
    # Repository in owner/repo format
    repo_url = "sympy/sympy"
    commit_hash = "8dcb12a6cf500e8738d6729ab954a261758f49ca" # Leave blank for latest commit

    chunks, cost_estimate = estimate_chunking_cost(repo_url, commit_hash)

# TODO: account for prompt caching