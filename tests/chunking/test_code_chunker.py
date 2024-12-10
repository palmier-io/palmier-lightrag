import pytest
from lightrag.chunking.code_chunker import CodeChunker, CodeChunk, Position, ChunkType
import os
import tempfile


class TestCodeChunker:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def chunker(self, temp_dir):
        return CodeChunker(root_dir=temp_dir, target_tokens=100)

    def test_chunk_simple_python_file(self, temp_dir, chunker):
        # Create a test Python file
        test_code = """def function1():
    print("Hello")

def function2():
    print("World")"""
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w") as f:
            f.write(test_code)

        chunks = chunker.chunk_file(file_path, test_code)

        assert len(chunks) == 1
        assert chunks[0]["content"] == test_code
        assert chunks[0]["tag"]["language"] == "python"

    def test_merge_chunks(self, chunker):
        content_1 = "def func1():\n    pass\ndef func2():\n    pass\n"
        content_2 = "def func3():\n    pass\ndef func4():\n    pass\n"
        full_content = content_1 + content_2

        token_1 = chunker.encoding.encode(content_1)
        token_2 = chunker.encoding.encode(content_2)
        token_full = chunker.encoding.encode(full_content)

        chunks = [
            CodeChunk(
                index=0,
                file_path="test.py",
                content=content_1,
                token_count=len(token_1),
                tag={"language": "python"},
                start=Position(line=0, character=0, byte=0),
                end=Position(
                    line=len(content_1.split("\n")), character=0, byte=len(token_1)
                ),
                chunk_type=ChunkType.TREE_SITTER,
            ),
            CodeChunk(
                index=1,
                file_path="test.py",
                content=content_2,
                token_count=len(token_2),
                tag={"language": "python"},
                start=Position(
                    line=len(content_1.split("\n")), character=0, byte=len(token_1)
                ),
                end=Position(
                    line=len(full_content.split("\n")),
                    character=0,
                    byte=len(token_full),
                ),
                chunk_type=ChunkType.TREE_SITTER,
            ),
        ]

        merged = chunker._merge_chunks(chunks, full_content)
        assert len(merged) == 1
        print(merged[0].content)
        print(full_content)
        assert merged[0].content == full_content
        assert merged[0].token_count == len(token_full)
        assert merged[0].chunk_type == ChunkType.MERGE

    def test_large_file_chunking(self, temp_dir, chunker):
        # Create a large Python file that should be split into multiple chunks
        large_function = "def large_function():\n" + "    print('x')\n" * 1000
        test_code = large_function * 3

        file_path = os.path.join(temp_dir, "large_test.py")
        with open(file_path, "w") as f:
            f.write(test_code)

        chunks = chunker.chunk_file(file_path, test_code)
        assert len(chunks) > 1
        assert all(
            chunk["token_count"] <= chunker.target_tokens + chunker.overlap_token_size
            for chunk in chunks
        )
