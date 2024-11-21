import os
from tree_sitter_languages import get_parser
from tree_sitter import Node
import tiktoken
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from .language_parsers import (
    get_language_from_file,
    SUPPORT_LANGUAGES,
    FILES_TO_IGNORE,
)


class ChunkType(Enum):
    TREE_SITTER = "tree_sitter"
    TOKEN_SIZE = "token_size"
    MERGE = "merge"


@dataclass
class Position:
    """A start/end position of a chunk"""

    # The byte offset of the position - Present for token-size
    byte: Optional[int] = None

    # The line number of the position - Present for tree-sitter
    line: Optional[int] = None

    # The character number within the line - Present for tree-sitter
    character: Optional[int] = None


@dataclass
class NodeInfo:
    """Information about a tree-sitter node"""

    text: str
    token_count: int
    node: Node


@dataclass
class CodeChunk:
    """Data class for a code chunk"""

    # The index of the chunk in the file
    index: int

    # The relative path to the file
    file_path: str

    # The content of the chunk
    content: str

    # The number of tokens in the chunk
    token_count: int

    # The start position of the chunk
    start: Position

    # The end position of the chunk
    end: Position

    # The type of chunking used
    chunk_type: ChunkType

    # Any metadata about the chunk
    tag: Optional[Dict[str, Any]] = None

    # Add encoding as a non-dataclass field
    def __post_init__(self):
        self._encoding = None

    @classmethod
    def create(
        cls, file_path: str, language_name: str, code_str: str, encoding
    ) -> "CodeChunk":
        """Create a new empty chunk"""
        chunk = cls(
            index=0,  # Will be set later
            file_path=file_path,
            content="",
            token_count=0,
            start=None,
            end=None,
            chunk_type=ChunkType.TREE_SITTER,
            tag={"language": language_name},
        )
        chunk._encoding = encoding
        return chunk

    def has_content(self) -> bool:
        """Check if the chunk has any content"""
        return self.start is not None

    def add_node(self, node_info: NodeInfo, code_str: str) -> None:
        """Add a node to the current chunk"""
        if not self.has_content():
            self.start = Position(
                line=node_info.node.start_point[0],
                character=node_info.node.start_point[1],
                byte=node_info.node.start_byte,
            )
        self.end = Position(
            line=node_info.node.end_point[0],
            character=node_info.node.end_point[1],
            byte=node_info.node.end_byte,
        )
        self.content = "\n".join(
            code_str.split("\n")[self.start.line : self.end.line + 1]
        )
        self.token_count = len(self._encoding.encode(self.content))

    def set_index(self, index: int) -> None:
        """Set the chunk index"""
        self.index = index

    def to_dict(self) -> Dict[str, Any]:
        """Convert CodeChunk to a dictionary"""
        return {
            "index": self.index,
            "file_path": self.file_path,
            "content": self.content,
            "token_count": self.token_count,
            "start": {
                "line": self.start.line,
                "character": self.start.character,
                "byte": self.start.byte,
            }
            if self.start
            else None,
            "end": {
                "line": self.end.line,
                "character": self.end.character,
                "byte": self.end.byte,
            }
            if self.end
            else None,
            "chunk_type": self.chunk_type.value,
            "tag": self.tag,
        }


class CodeChunker:
    def __init__(
        self,
        root_dir,
        target_tokens=800,
        overlap_token_size=128,
        tiktoken_model="gpt-4o",
    ):
        # Local root directory of where the repo is downloaded to
        self.root_dir = root_dir

        # Target tokens per chunk
        self.target_tokens = target_tokens

        # Overlap tokens between chunks
        self.overlap_token_size = overlap_token_size

        # Encoding to calculate token count
        self.encoding = tiktoken.encoding_for_model(tiktoken_model)

    def chunk_file(self, full_file_path: str) -> List[Dict[str, Any]]:
        """Given a full file path, return a list of chunks as dictionaries."""

        relative_file_path = full_file_path.replace(self.root_dir, "")
        # Remove leading separator and split path
        relative_file_path = relative_file_path.lstrip(os.sep)
        # Remove first folder from path - this is the zip folder name downloaded from GitHub
        # relative_file_path = os.sep.join(relative_file_path.split(os.sep)[1:])

        if any(relative_file_path.endswith(ext) for ext in FILES_TO_IGNORE):
            logging.debug(f"Skipping file {relative_file_path}")
            return []

        with open(full_file_path, "rb") as f:
            content_bytes = f.read()

        language_name = get_language_from_file(full_file_path)

        if language_name == "text only":
            content = content_bytes.decode("utf-8", errors="ignore")
            chunks = self._chunking_by_token_size(content, relative_file_path)
        elif language_name in SUPPORT_LANGUAGES:
            chunks = self._chunking_by_tree_sitter(
                content_bytes, language_name, relative_file_path
            )
        else:
            logging.debug(
                f"Skipping file {full_file_path} - Unsupported language: {language_name}"
            )
            return []

        return [chunk.to_dict() for chunk in chunks]

    def _chunking_by_token_size(
        self, content: str, file_path: str, current_index: int = 0
    ) -> List[CodeChunk]:
        """Chunk the content of a file by token size"""

        tokens = self.encoding.encode(content)
        results = []

        for index, start in enumerate(
            range(0, len(tokens), self.target_tokens - self.overlap_token_size)
        ):
            end = start + self.target_tokens
            chunk_content = self.encoding.decode(tokens[start:end])

            results.append(
                CodeChunk(
                    index=current_index + index,
                    file_path=file_path,
                    content=chunk_content.strip(),
                    token_count=min(self.target_tokens, len(tokens) - start),
                    start=Position(byte=start),
                    end=Position(byte=end),
                    chunk_type=ChunkType.TOKEN_SIZE,
                )
            )
        return results

    def _chunking_by_tree_sitter(
        self, content_bytes: bytes, language_name: str, file_path: str
    ) -> List[CodeChunk]:
        """Chunk the content of a file using tree-sitter"""

        try:
            parser = get_parser(language_name)
        except LookupError:
            # Bug
            raise ValueError(f"Parser not found for language: {language_name}")

        tree = parser.parse(content_bytes)

        code_str = content_bytes.decode("utf-8", errors="ignore")
        current_index = 0

        def traverse(node: Node) -> List[CodeChunk]:
            """Recursively traverse the tree-sitter tree"""
            nonlocal code_str, language_name, file_path, current_index

            if len(node.children) == 0:
                text = code_str[node.start_byte : node.end_byte]
                return self._chunking_by_token_size(text, file_path, current_index)

            new_chunks: List[CodeChunk] = []
            current_chunk = CodeChunk.create(
                file_path, language_name, code_str, self.encoding
            )

            for child in node.children:
                child_info = NodeInfo(
                    text=code_str[child.start_byte : child.end_byte],
                    token_count=len(
                        self.encoding.encode(
                            code_str[child.start_byte : child.end_byte]
                        )
                    ),
                    node=child,
                )

                # The child node itself is too big, so we need to recursively traverse the child nodes
                if self._should_traverse_child(child_info.token_count):
                    if current_chunk.has_content():
                        current_chunk.set_index(current_index)
                        new_chunks.append(current_chunk)
                        current_index += 1

                    new_chunks.extend(traverse(child))
                    current_chunk = CodeChunk.create(
                        file_path, language_name, code_str, self.encoding
                    )

                # The child node does not fit in the current chunk, so we need to start a new chunk
                elif self._should_start_new_chunk(
                    current_chunk.token_count, child_info.token_count
                ):
                    if current_chunk.has_content():
                        current_chunk.set_index(current_index)
                        new_chunks.append(current_chunk)
                        current_index += 1

                    current_chunk = CodeChunk.create(
                        file_path, language_name, code_str, self.encoding
                    )
                    current_chunk.add_node(child_info, code_str)

                else:
                    current_chunk.add_node(child_info, code_str)

            if current_chunk.has_content():
                current_chunk.set_index(current_index)
                new_chunks.append(current_chunk)
                current_index += 1

            return new_chunks

        # Start chunking at the root node
        chunks = traverse(tree.root_node)

        # Merge small chunks
        merged_chunks = self._merge_chunks(chunks, code_str)

        return merged_chunks

    def _merge_chunks(self, chunks: List[CodeChunk], code_str: str) -> List[CodeChunk]:
        """
        Merge small chunks together while respecting target_tokens limit,
        starting from the end of the list to preserve context.
        """
        if not chunks:
            return []

        # Work with reversed list
        chunks = chunks[::-1]
        merged = []
        current_chunk = chunks[0]

        for last_chunk in chunks[1:]:
            # Can only merge Tree-Sitter chunks
            if (
                current_chunk.chunk_type == ChunkType.TOKEN_SIZE
                or last_chunk.chunk_type == ChunkType.TOKEN_SIZE
            ):
                merged.append(current_chunk)
                current_chunk = last_chunk
                continue

            # If merging wouldn't exceed target_tokens, combine the chunks
            if (
                current_chunk.token_count + last_chunk.token_count
                <= self.target_tokens + self.overlap_token_size
            ):
                # Read directly from code_str to get the merged content
                content = "\n".join(
                    code_str.split("\n")[
                        last_chunk.start.line : current_chunk.end.line + 1
                    ]
                )

                current_chunk = CodeChunk(
                    index=last_chunk.index,  # Use the earlier index
                    file_path=last_chunk.file_path,
                    start=last_chunk.start,
                    end=current_chunk.end,
                    content=content,
                    token_count=len(
                        self.encoding.encode(content)
                    ),  # Recalculate tokens
                    tag=last_chunk.tag,
                    chunk_type=ChunkType.MERGE,
                )
                current_chunk._encoding = self.encoding
            else:
                # Can't merge anymore, add current_chunk to results and start new chunk
                merged.append(current_chunk)
                current_chunk = last_chunk

        # Don't forget to add the last chunk
        merged.append(current_chunk)

        # Reverse back to original order and update indices
        merged = merged[::-1]
        for i, chunk in enumerate(merged):
            chunk.index = i

        return merged

    def _should_traverse_child(self, token_count: int) -> bool:
        """Determine if a child node should be traversed recursively"""
        return token_count > self.target_tokens

    def _should_start_new_chunk(self, current_tokens: int, new_tokens: int) -> bool:
        """Determine if we should start a new chunk"""
        return current_tokens + new_tokens > self.target_tokens


def traverse_directory(root_dir: str) -> List[str]:
    """
    Walk the directory and return a list of full file paths. Ignore files that are not supported.
    """

    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in FILES_TO_IGNORE):
                continue

            language_name = get_language_from_file(os.path.join(root, file))
            if language_name != "text only" and language_name not in SUPPORT_LANGUAGES:
                continue

            file_list.append(os.path.join(root, file))
    return file_list


# NOT USED
def generate_file_summary(
    code_str: str, file_path: str, model: str = "gpt-4o-mini"
) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Please provide a high-level summary of the given code content in file {file_path}. Include key concepts and functionalities, mentioning relevant classes and function names along with their purposes and interactions. Take into account the file path name for context. Keep the summary concise, using no more than 100 words, and format it as a single paragraph.",
            },
            {"role": "user", "content": code_str},
        ],
    )
    return response.choices[0].message.content
