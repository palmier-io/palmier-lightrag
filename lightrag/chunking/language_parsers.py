import os
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Only chunk files with these extensions - allow direct chunk search
CHUNKING_SUPPORT_LANGUAGES = [
    "bash",
    "c",
    "c_sharp",
    "commonlisp",
    "cpp",
    "css",
    "dockerfile",
    "dot",
    "elisp",
    "elixir",
    "elm",
    "erlang",
    "go",
    "gomod",
    "hack",
    "haskell",
    "hcl",
    "html",
    "java",
    "javascript",
    "jsdoc",
    "json",
    "julia",
    "kotlin",
    "lua",
    "make",
    "markdown",
    "objc",
    "ocaml",
    "perl",
    "php",
    "python",
    "ql",
    "r",
    "regex",
    "rst",
    "ruby",
    "rust",
    "scala",
    "sql",
    "sqlite",
    "tsq",
    "typescript",
    "tsx",
    "yaml",
]

# Only extract entities from these extensions - reduce graph size and focus on code
EXTRACT_ENTITIES_SUPPORT_LANGUAGES = [
    "c",
    "c_sharp",
    "cpp",
    "go",
    "java",
    "javascript",
    "kotlin",
    "lua",
    "objc",
    "ocaml",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "typescript",
    "tsx",
]

FILES_TO_IGNORE = [
    # General lock files in JSON/YAML formats
    ".lock",
    "lock.json",
    "lock.yaml",
    "lock.yml",
    "__init__.py",
    ".json",
    ".xml",
]

FOLDERS_TO_IGNORE = [
    ".git",
    ".github",
    ".vscode",
    ".DS_Store",
    ".venv",
    ".pytest_cache",
    "__pycache__",
]


def should_ignore_file(file_path: str) -> bool:
    path = Path(file_path)
    if any(folder in path.parts for folder in FOLDERS_TO_IGNORE):
        return True
    if path.is_file():
        if any(path.name.endswith(ext) for ext in FILES_TO_IGNORE):
            return True
        language = get_language_from_file(file_path)
        if language != "text only" and language not in CHUNKING_SUPPORT_LANGUAGES:
            return True
    return False


def get_language_from_file(file_path) -> Optional[str]:
    """
    Given a file path, extract the extension and use pygment lexer to identify the language.
    tsx is a special case, so we handle it separately.
    https://pygments.org/languages/
    """

    # Handle special case for tsx
    extension = os.path.splitext(file_path)[1]
    if extension == ".tsx":
        return "tsx"

    try:
        lexer = get_lexer_for_filename(file_path)
        name = lexer.name.lower()

        # Handle special cases
        match name:
            case "c++":
                return "cpp"
            case "docker":
                return "dockerfile"
            case _:
                return name
    except ClassNotFound:
        return None
