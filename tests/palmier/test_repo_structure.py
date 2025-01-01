import pytest
from pathlib import Path
from lightrag.palmier.repo_structure import generate_directory_tree, generate_skeleton, check_ast_grep_installed

# Setup test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

def test_directory_tree():
    """Test directory tree generation"""
    tree = generate_directory_tree(str(TEST_DATA_DIR))
    print(tree)
    assert "── java" in tree
    assert "── Example.java" in tree
    assert "── javascript" in tree
    assert "── example.js" in tree
    assert "── python" in tree
    assert "── example.py" in tree
    assert "── typescript" in tree
    assert "── example.ts" in tree

def test_python_skeleton():
    """Test Python file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")
        
    python_file = TEST_DATA_DIR / "python" / "example.py"
    skeleton = generate_skeleton(str(python_file))
    
    # Verify expected structure
    assert "class ExampleClass:" in skeleton
    assert "def __init__(self, name: str):" in skeleton
    assert '"""Initialize with name."""' in skeleton
    assert "def process_data(self, data: list) -> dict:" in skeleton

def test_javascript_skeleton():
    """Test JavaScript file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")
        
    js_file = TEST_DATA_DIR / "javascript" / "example.js"
    skeleton = generate_skeleton(str(js_file))
    print(skeleton)
    # Verify expected structure
    assert "class DataProcessor {" in skeleton
    assert "constructor(name) {" in skeleton
    assert "processData(data) {" in skeleton

def test_typescript_skeleton():
    """Test TypeScript file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")
        
    ts_file = TEST_DATA_DIR / "typescript" / "example.ts"
    skeleton = generate_skeleton(str(ts_file))
    
    # Verify expected structure
    assert "interface DataProcessor {" in skeleton
    assert "class DataProcessorImpl implements DataProcessor {" in skeleton
    assert "constructor(private name: string) {" in skeleton

def test_java_skeleton():
    """Test Java file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")
        
    java_file = TEST_DATA_DIR / "java" / "Example.java"
    skeleton = generate_skeleton(str(java_file))
    
    # Verify expected structure
    assert "public class Example {" in skeleton
    assert "public Example(String name) {" in skeleton
    assert "public Map<String, Object> processData(List<String> data) {" in skeleton
