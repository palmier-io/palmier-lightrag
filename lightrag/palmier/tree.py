import os
import logging
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass
import time

from ..base import BaseVectorStorage
from ..utils import compute_mdhash_id
from ..chunking.language_parsers import should_ignore_file
from ..prompt import PROMPTS
logger = logging.getLogger(__name__)

class SummaryType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    GLOBAL = "global"

@dataclass
class SummaryNode:
    id: str
    type: SummaryType
    content: str
    file_path: str
    children: Dict[str, 'SummaryNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @staticmethod
    async def create_file_node(
        path: str,
        root_directory: str,
        tree: str,
        use_llm_func: callable,
        max_content_length: int = 2000
    ) -> Optional['SummaryNode']:
        """Create a SummaryNode for a file."""

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_content_length)
        relative_path = os.path.relpath(path, root_directory)

        prompt = PROMPTS["file_summary"].format(
            tree=tree,
            path=relative_path,
            content=content
        )
        
        summary = await use_llm_func(prompt, max_tokens=200)
        logger.debug(f"Generated summary for {relative_path}: {summary}")

        return SummaryNode(
            id=compute_mdhash_id(relative_path, prefix="sum-"),
            type=SummaryType.FILE,
            content=summary,
            file_path=relative_path
        )
    
    @staticmethod
    async def create_directory_node(
        path: str,
        root_directory: str,
        tree: str,
        children_nodes: List['SummaryNode'],
        use_llm_func: callable
    ) -> 'SummaryNode':
        """Create a SummaryNode for a directory using its children's summaries."""
        relative_path = os.path.relpath(path, root_directory)
        if not children_nodes:
            return SummaryNode(
                id=compute_mdhash_id(relative_path, prefix="sum-"),
                type=SummaryType.DIRECTORY,
                content="Empty directory",
                file_path=relative_path
            )
            
        children_summaries = [node.content for node in children_nodes]
        prompt = PROMPTS["folder_summary"].format(
            tree=tree,
            path=relative_path,
            children_summaries=children_summaries
        )
        
        summary = await use_llm_func(prompt, max_tokens=200)
        node = SummaryNode(
            id=compute_mdhash_id(relative_path, prefix="sum-"),
            type=SummaryType.DIRECTORY,
            content=summary,
            file_path=relative_path
        )
        logger.debug(f"Generated summary for {relative_path}: {summary}")
        
        # Add children to the node
        for child in children_nodes:
            node.children[child.file_path] = child
            
        return node
    
    async def create_global_node(
        root_directory: str,
        tree: str,
        children_nodes: List['SummaryNode'],
        use_llm_func: callable
    ) -> 'SummaryNode':
        """Create a SummaryNode for the root directory."""

        readme_path = os.path.join(root_directory, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                readme_content = f.read()
        else:
            readme_content = ""

        joined_summaries = "\n".join([node.content for node in children_nodes])

        prompt = PROMPTS["global_summary"].format(
            readme_content=readme_content,
            tree=tree,
            joined_summaries=joined_summaries
        )

        summary = await use_llm_func(prompt, max_tokens=500)
        logger.debug(f"Generated global summary: {summary}")
        return SummaryNode(
            id=compute_mdhash_id("/", prefix="sum-"),
            type=SummaryType.GLOBAL,
            content=summary,
            file_path='/'
        )

    def to_dict(self) -> dict:
        """Convert the node to a dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "file_path": self.file_path,
            "type": self.type.value 
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SummaryNode':
        """Create a SummaryNode from a dictionary."""
        return cls(
            id=data["id"],
            type=SummaryType(data["type"]),
            content=data["content"],
            file_path=data["file_path"]
        )
    
    @staticmethod
    async def get_existing_node(path: str, summaries: BaseVectorStorage) -> Optional['SummaryNode']:
        """Fetch an existing node from vector storage."""
        node_id = compute_mdhash_id(path, prefix="sum-")
        results = await summaries.query_by_id(node_id)
        if results:
            return SummaryNode.from_dict(results)
        return None

async def update_summary_recursive(
    path: str,
    root_directory: str,
    tree: str,
    summaries: BaseVectorStorage,
    use_llm_func: callable,
    exclude_paths: set[str] = None,
    updates: dict[str, dict] = None
) -> Optional[SummaryNode]:
    """
    Recursively update summaries for a path and its children.
    Returns a SummaryNode for the path.
    """
    exclude_paths = exclude_paths or set()
    path_obj = Path(path)
    
    # Return node if no update is needed
    if str(path_obj) in exclude_paths:
        return await SummaryNode.get_existing_node(str(path_obj), summaries)

    if should_ignore_file(path):
        return None

    # Bottom-up - create file nodes first
    if path_obj.is_file():
        node = await SummaryNode.create_file_node(str(path_obj), root_directory, tree, use_llm_func)            
        if node:
            updates[node.id] = node.to_dict()
        return node

    children_nodes = []
    try:
        for entry in path_obj.iterdir():
            child_node = await update_summary_recursive(
                str(entry),
                root_directory,
                tree,
                summaries,
                use_llm_func,
                exclude_paths,
                updates
            )
            if child_node:
                children_nodes.append(child_node)
    except (PermissionError, OSError) as e:
        logger.warning(f"Error accessing directory {path}: {e}")
        return None

    # Create directory node
    dir_node = await SummaryNode.create_directory_node(
        str(path_obj),
        root_directory,
        tree,
        children_nodes,
        use_llm_func
    )
    
    # Store in vector database
    updates[dir_node.id] = dir_node.to_dict()
    
    return dir_node

async def update_summary(
    directory: str,
    file_paths: list[str],
    summaries: BaseVectorStorage,
    use_llm_func: callable
):
    """
    Update summaries in the vector database for specified files and their parent directories.
    """
    start_time = time.time()
    
    # Get all paths that don't need updating
    exclude_paths = set()
    file_paths_set = set(file_paths)
    
    # Also include parent directories of files that need updating
    parent_paths = set()
    for file_path in file_paths:
        current = Path(file_path).parent
        while str(current) >= str(directory):
            parent_paths.add(str(current))
            current = current.parent
    
    # Add to exclude_paths only if not a parent of updated files
    for path in Path(directory).rglob('*'):
        path_str = str(path)
        if path_str not in file_paths_set and path_str not in parent_paths:
            exclude_paths.add(path_str)
    
    tree = generate_directory_tree(directory)
    updates = {}
    root_node = await update_summary_recursive(
        directory,
        directory,
        tree,
        summaries,
        use_llm_func,
        exclude_paths,
        updates
    )

    if root_node:
        global_node = await SummaryNode.create_global_node(
            directory,
            tree,
            list(root_node.children.values()),
            use_llm_func
        )
        updates[global_node.id] = global_node.to_dict()

    if updates:
        logger.info(f"Batch updating {len(updates)} summaries")
        await summaries.upsert(updates)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"[Update Summary] Completed in {duration:.2f} seconds")
    
    return root_node

def generate_directory_tree(root_directory: str, prefix: str = "", level: int = -1) -> str:
    """
    Generate a directory tree recursively in string format
    Filters out files/folders that should be ignored based on should_ignore_file.

    Args:
        root_directory: Root path to start tree generation
        prefix: Prefix for the current line (used for recursion)
        level: Maximum depth to traverse (-1 for unlimited)

    Example:
    ```
    root
    ├── src
    │   ├── main.py
    │   └── subdir
    │       ├── module.py
    │       └── utils.py
    └── tests
        └── test_main.py
    ```
    """
    directory = Path(root_directory)
    result = []
    
    if level == 0:
        return ""

    try:
        # Get all contents and filter out ignored files/folders
        contents = []
        for item in sorted(directory.iterdir()):
            if should_ignore_file(str(item)):
                continue
            contents.append(item)
    except PermissionError:
        return f"{prefix}├── [Permission Denied]\n"

    total = len(contents)

    for index, item in enumerate(contents, start=1):
        is_last = index == total
        connector = "└── " if is_last else "├── "
                
        result.append(f"{prefix}{connector}{item.name}")
        
        # If it's a directory, recursively process its contents
        if item.is_dir():
            subtree = generate_directory_tree(
                str(item),
                prefix + ("    " if is_last else "│   "),
                level - 1 if level != -1 else -1
            )
            if subtree:
                result.append(subtree)
    
    return "\n".join(result)