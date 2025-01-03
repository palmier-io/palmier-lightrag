import pytest
from pathlib import Path
from lightrag.palmier.repo_structure import (
    generate_directory_tree,
    generate_skeleton,
    check_ast_grep_installed,
)

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
    # Verify expected structure
    assert "class DataProcessor {" in skeleton
    assert "constructor(name) {" in skeleton
    assert "processData(data) {" in skeleton

    # Test JSX file
    jsx_file = TEST_DATA_DIR / "javascript" / "example.jsx"
    jsx_skeleton = generate_skeleton(str(jsx_file))
    # Verify JSX structure
    assert "import React, { useState, useEffect } from 'react';" in jsx_skeleton
    assert "const DataProcessorComponent = ({ initialName }) => {" in jsx_skeleton
    assert "export default DataProcessorComponent;" in jsx_skeleton
    assert "// State management with hooks" in jsx_skeleton


def test_typescript_skeleton():
    """Test TypeScript file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")

    ts_file = TEST_DATA_DIR / "typescript" / "example.ts"
    skeleton = generate_skeleton(str(ts_file))

    # Verify expected structure
    assert "interface DataProcessor {" in skeleton
    assert "class DataProcessorImpl implements DataProcessor {" in skeleton
    assert "constructor(name: string) {" in skeleton

    # Test TSX file
    tsx_file = TEST_DATA_DIR / "typescript" / "example.tsx"
    tsx_skeleton = generate_skeleton(str(tsx_file))

    # Verify TSX structure
    assert "import React, { useState, useEffect } from 'react';" in tsx_skeleton
    assert "interface DataProcessor {" in tsx_skeleton
    assert "const DataProcessorComponent: React.FC<{" in tsx_skeleton
    assert (
        "const processData = (inputData: string[]): Record<string, number> => {"
        in tsx_skeleton
    )
    assert "export default DataProcessorComponent;" in tsx_skeleton


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


def test_cpp_skeleton():
    """Test C++ file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")

    cpp_file = TEST_DATA_DIR / "cpp" / "example.cpp"
    skeleton = generate_skeleton(str(cpp_file))

    # Verify expected structure
    assert "#include <string>" in skeleton
    assert "namespace processor {" in skeleton
    assert "// Enum for processing status" in skeleton
    assert "enum class ProcessStatus {" in skeleton
    assert "class IProcessor {" in skeleton
    assert "class DataProcessor : public IProcessor {" in skeleton
    assert "DataProcessor(const std::string& name)" in skeleton
    assert "std::map<std::string, int> processData" in skeleton
    assert "ProcessStatus getStatus() const" in skeleton


def test_golang_skeleton():
    """Test Go file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")

    go_file = TEST_DATA_DIR / "golang" / "example.go"
    skeleton = generate_skeleton(str(go_file))

    # Verify expected structure
    assert "package processor" in skeleton
    assert "type ProcessStatus int" in skeleton
    assert "type ProcessError string" in skeleton
    assert "func (e ProcessError) Error() string {" in skeleton
    assert "type DataProcessor interface {" in skeleton
    assert "type DefaultProcessor struct {" in skeleton
    assert "func NewDefaultProcessor(name string) *DefaultProcessor {" in skeleton
    assert (
        "func (dp *DefaultProcessor) ProcessData(data []string) (map[string]interface{}, error) {"
        in skeleton
    )
    assert "func (dp *DefaultProcessor) GetStatus() ProcessStatus {" in skeleton


def test_rust_skeleton():
    """Test Rust file skeleton generation"""
    if not check_ast_grep_installed():
        pytest.skip("ast-grep not installed")

    rust_file = TEST_DATA_DIR / "rust" / "example.rs"
    skeleton = generate_skeleton(str(rust_file))

    # Verify expected structure
    assert "use std::collections::HashMap;" in skeleton
    assert "/// ProcessStatus represents the status of data processing" in skeleton
    assert "pub enum ProcessStatus {" in skeleton
    assert "pub struct ProcessError(String);" in skeleton
    assert "impl Error for ProcessError {}" in skeleton
    assert "pub trait DataProcessor {" in skeleton
    assert "pub struct DefaultProcessor {" in skeleton
    assert "impl DefaultProcessor {" in skeleton
    assert "impl DataProcessor for DefaultProcessor {" in skeleton
    assert "pub struct GenericProcessor<T> {" in skeleton
    assert "impl<T> GenericProcessor<T> {" in skeleton
    assert "pub fn combine(first: String, second: String) -> String {" in skeleton
    assert "pub const MAX_RETRIES: u32 = 3;" in skeleton