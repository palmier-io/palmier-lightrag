import os
import ast
from typing import List, Dict, Any
from utils.ast_parser import parse_ast
import tiktoken
from tqdm import tqdm

class CodebaseChunker:
    def __init__(self, codebase_path):
        self.codebase_path = codebase_path
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 1024  # You can adjust this value as needed

    def chunk_codebase(self):
        chunks = []
        if not os.path.exists(self.codebase_path):
            raise FileNotFoundError(f"Codebase directory not found: {self.codebase_path}")
        
        python_files = [os.path.join(root, file) 
                        for root, _, files in os.walk(self.codebase_path) 
                        for file in files if file.endswith('.py')]
        
        for file_path in tqdm(python_files, desc="Chunking codebase", unit="file"):
            print(f"Chunking: {file_path}")  # Print each file path
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                ast_tree = parse_ast(content)
                file_chunks = self._generate_chunks(ast_tree, file_path)
                chunks.extend(file_chunks)
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error processing file {file_path}: {e}")
        return chunks

    def _generate_chunks(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        chunks = []
        
        # Function-level chunks
        function_chunks = self._get_function_chunks(tree, file_path)
        chunks.extend(function_chunks)
        
        # File-level chunk
        file_chunk = self._get_file_chunk(tree, file_path)
        chunks.append(file_chunk)

        return chunks

    def _get_function_chunks(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        function_chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunk = self._create_chunk(node, file_path)
                if chunk:
                    function_chunks.append(chunk)
        return function_chunks

    def _get_file_chunk(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        return {
            'type': 'file',
            'content': ast.unparse(tree),
            'metadata': {
                'file': file_path,
                'name': os.path.basename(file_path)
            }
        }

    def _create_chunk(self, node: ast.AST, file_path: str) -> Dict[str, Any]:
        content = ast.unparse(node)
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= self.max_tokens:
            return {
                'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                'content': content,
                'metadata': {
                    'file': file_path,
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno
                }
            }
        else:
            # If the chunk is too large, we need to split it
            return self._split_large_chunk(node, file_path)

    def _split_large_chunk(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        chunks = []
        content = ast.unparse(node)
        tokens = self.tokenizer.encode(content)
        
        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_content = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'type': 'function_part' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class_part',
                'content': chunk_content,
                'metadata': {
                    'file': file_path,
                    'name': f"{node.name}_part_{len(chunks) + 1}",
                    'original_name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno
                }
            })
            
            start = end
        
        return chunks

    def _print_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            print("\n--- Chunk ---")
            print(f"Type: {chunk['type']}")
            print(f"Metadata: {chunk['metadata']}")
            print("Content:")
            print(chunk['content'])
            print("--- End Chunk ---\n")

    def get_multi_file_chunks(self, max_tokens: int = 4096) -> List[Dict[str, Any]]:
        all_chunks = self.chunk_codebase()
        multi_file_chunks = []
        current_chunk = {'type': 'multi_file', 'content': '', 'metadata': {'files': []}}
        current_tokens = 0

        for chunk in all_chunks:
            chunk_tokens = len(chunk['content'].split())
            if current_tokens + chunk_tokens > max_tokens:
                if current_chunk['content']:
                    multi_file_chunks.append(current_chunk)
                current_chunk = {'type': 'multi_file', 'content': '', 'metadata': {'files': []}}
                current_tokens = 0

            current_chunk['content'] += chunk['content'] + '\n\n'
            current_chunk['metadata']['files'].append(chunk['metadata']['file'])
            current_tokens += chunk_tokens

        if current_chunk['content']:
            multi_file_chunks.append(current_chunk)

        return multi_file_chunks
