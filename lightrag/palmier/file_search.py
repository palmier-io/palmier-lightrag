from pathlib import Path
from typing import List, Tuple
from thefuzz import fuzz

def fuzzy_file_search(file_paths: List[str], base_dir: Path, threshold: int = 75) -> List[Tuple[str, float]]:
    """
    Find matching files from a list of potential file paths, supporting glob patterns.
    
    Args:
        file_paths: List of potential file paths (can include glob patterns)
        base_dir: Base directory to search for files
        threshold: Minimum similarity score (0-100)

    Returns:
        List of tuples containing (file_path, confidence_score)
    """
    results = []
    existing_files = list(base_dir.rglob("*"))
    
    for path_pattern in file_paths:
        # Handle glob patterns directly
        if '*' in path_pattern:
            glob_pattern = base_dir / Path(path_pattern)
            matching_files = [
                str(f.relative_to(base_dir)) 
                for f in base_dir.glob(path_pattern)
                if f.is_file()
            ]
            # Add exact matches from glob with 100% confidence
            results.extend((path, 1.0) for path in matching_files)
            continue
            
        # For non-glob patterns, do fuzzy matching
        best_match = None
        best_score = 0
        path_lower = path_pattern.lower()
        
        for file_path in existing_files:
            if not file_path.is_file():
                continue
                
            rel_path = str(file_path.relative_to(base_dir))
            rel_path_lower = rel_path.lower()
            
            partial_score = fuzz.partial_ratio(path_lower, rel_path_lower)
            
            if partial_score >= threshold:
                token_score = fuzz.token_sort_ratio(path_lower, rel_path_lower)
                score = max(partial_score, token_score)
            else:
                score = partial_score
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = rel_path
        
        if best_match:
            results.append((best_match, best_score / 100.0))
    
    # Sort by confidence score
    results.sort(key=lambda x: x[1], reverse=True)
    return results