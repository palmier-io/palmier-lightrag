import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tempfile
import subprocess
from lightrag.chunking.code_chunker import CodeChunker
from lightrag.utils import logger
from lightrag.prompt import PROMPTS
from lightrag.utils import encode_string_by_tiktoken
from dotenv import load_dotenv

load_dotenv()

def clone_repo(repo_url: str, temp_dir: str) -> str:
    """Clone a GitHub repository to a temporary directory."""
    logger.info(f"Cloning {repo_url} to {temp_dir}")
    
    # For HTTPS authentication
    if repo_url.startswith("https://"):
        # Use personal access token if available
        token = os.getenv("GITHUB_TOKEN")
        if token:
            repo_url = repo_url.replace("https://", f"https://{token}@")
    
    # For SSH authentication
    # Make sure your SSH key is added to the SSH agent
    subprocess.run(['git', 'clone', repo_url, temp_dir], check=True)
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
    
    for chunk in chunks:
        # Initial extraction call
        content_tokens = len(encode_string_by_tiktoken(chunk.get("content", "")))
        total_input_tokens += content_tokens + base_prompt_tokens
        total_output_tokens += 1000  # Estimate for entity/relation extraction TODO: find better estimate
        total_api_calls += 1

        previous_context_tokens = content_tokens + base_prompt_tokens + 1000
        
        # Estimate gleaning calls
        for _ in range(MAX_GLEANING):
            # Continue prompt with previous context
            total_input_tokens += continue_prompt_tokens + previous_context_tokens  # Previous context estimate
            total_output_tokens += 500  # Additional entities/relations TODO: find better estimate
            total_api_calls += 1
            previous_context_tokens += continue_prompt_tokens + 500 # Additional entities/relations
            
            # If-loop prompt with previous context
            total_input_tokens += if_loop_prompt_tokens + previous_context_tokens 
            total_output_tokens += 10  # Yes/No response
            total_api_calls += 1
    
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
    total_api_calls += merge_api_calls
    
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
        "prompt_token_counts": {
            "base_prompt": base_prompt_tokens,
            "continue_prompt": continue_prompt_tokens,
            "if_loop_prompt": if_loop_prompt_tokens,
            "summary_prompt": summary_prompt_tokens
        },
        "merge_costs": {
            "api_calls": merge_api_calls,
            "input_tokens": merge_input_tokens,
            "output_tokens": merge_output_tokens,
            "input_cost": round((merge_input_tokens / 1000000) * MODEL_COSTS[model_name]["input"], 2),
            "output_cost": round((merge_output_tokens / 1000000) * MODEL_COSTS[model_name]["output"], 2),
        }
    }

def estimate_chunking_cost(repo_url: str):
    """Estimate the cost of chunking and processing a repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone repo
        repo_dir = clone_repo(repo_url, temp_dir)
        
        # Initialize chunker
        chunker = CodeChunker(
            repo_dir,
            target_tokens=800,
            overlap_token_size=100,
            tiktoken_model="gpt-4o-mini"
        )
        
        # Get all files
        file_paths = get_all_files(repo_dir)
        
        # Collect chunks
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")
        
        # Print statistics
        logger.info(f"Total files processed: {len(file_paths)}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Estimate entity extraction costs
        cost_estimate = estimate_entity_extraction_cost(all_chunks)
        logger.info("Cost Estimation:")
        logger.info(f"Model: {cost_estimate['model']}")
        logger.info(f"Total API calls: {cost_estimate['total_api_calls']:,}")
        logger.info(f"Estimated input tokens: {cost_estimate['estimated_input_tokens']:,}")
        logger.info(f"Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
        logger.info(f"Estimated input cost: ${cost_estimate['estimated_input_cost']:.2f}")
        logger.info(f"Estimated output cost (varies significantly): ${cost_estimate['estimated_output_cost']:.2f}")
        logger.info(f"Estimated total cost: ${cost_estimate['estimated_total_cost']:.2f}")
        
        return all_chunks, cost_estimate

if __name__ == "__main__":
    # Example usage
    repo_url = "https://github.com/dylanwhawk/assist-llm"
    chunks, cost_estimate = estimate_chunking_cost(repo_url)

# TODO: account for prompt caching