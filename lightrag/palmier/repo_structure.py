import logging
from pathlib import Path
import subprocess
import json
from ..chunking.language_parsers import should_ignore_file, get_language_from_file

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


def check_ast_grep_installed() -> bool:
    """Check if ast-grep CLI is installed."""
    try:
        result = subprocess.run(["sg", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_skeleton(file_path: str) -> str:
    """
    Generate a skeleton structure of a file using ast-grep patterns.
    Returns the structural elements without implementation details.
    """
    if not check_ast_grep_installed():
        logger.error(
            "ast-grep CLI (sg) is not installed. Please install it with 'cargo install ast-grep'"
        )
        return ""

    file_path = str(Path(file_path))
    language = get_language_from_file(file_path)

    if not language:
        logger.warning(f"Unsupported file type: {file_path}")
        return ""

    language = "javascript" if language == "jsx" else language

    rules_dir = Path(__file__).parent / "ast_grep" / "rules" / f"{language}.yml"
    if not rules_dir.exists():
        logger.error(f"Rules file not found: {rules_dir}")
        return ""

    try:
        result = subprocess.run(
            ["sg", "scan", "--rule", str(rules_dir), "--json", file_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"ast-grep failed: {result.stderr}")
            return ""

        try:
            matches = json.loads(result.stdout)
            # Store tuples of (line_number, text)
            ordered_lines = []
            seen = set()

            for match in matches:
                lines = match.get("lines", "")
                line = lines.split("\n")[0]
                if line in seen:
                    continue
                seen.add(line)

                start_line = match.get("range", {}).get("start", {}).get("line", 0)
                ordered_lines.append((start_line, line))

            # Sort by line number and format
            ordered_lines.sort(key=lambda x: x[0])
            result_lines = [text for _, text in ordered_lines]

            return "\n".join(result_lines)

        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {result.stdout}")
            return ""

    except subprocess.CalledProcessError as e:
        logger.error(f"ast-grep failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error generating skeleton: {e}")
        return ""
