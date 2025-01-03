import logging
from pathlib import Path
from ..chunking.language_parsers import should_ignore_file, get_language_from_file
from .ast_grep.client import AstGrepClient

logger = logging.getLogger(__name__)


def generate_directory_tree(
    root_directory: str, prefix: str = "", level: int = -1
) -> str:
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
    if not directory.exists():
        logger.warning(f"Directory {root_directory} does not exist")
        return ""
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
                level - 1 if level != -1 else -1,
            )
            if subtree:
                result.append(subtree)

    return "\n".join(result)


def generate_skeleton(file_path: str) -> str:
    """
    Generate a skeleton structure of a file using ast-grep.
    Returns the structural elements without implementation details.
    """
    file_path = Path(file_path)
    language = get_language_from_file(str(file_path))

    if not language:
        logger.warning(f"Unsupported file type: {file_path}")
        return ""

    language = "javascript" if language == "jsx" else language

    client = AstGrepClient()
    return client.get_skeleton(str(file_path), language)
