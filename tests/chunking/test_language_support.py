import pytest
from lightrag.chunking.code_chunker import CodeChunker
import os
import tempfile
from tree_sitter_languages import get_parser

# Small target_tokens to force splitting
target_tokens = 50


class TestLanguageSupport:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    @pytest.fixture
    def chunker(self, temp_dir):
        # Reduced target_tokens to force splitting
        return CodeChunker(root_dir=temp_dir, target_tokens=target_tokens, overlap_token_size=10)

    def test_python_chunking(self, temp_dir, chunker):
        function_1 = """def function1():
    print("This is a")
    print("longer function")
    print("That should split into")
    print("Multiple chunks")"""

        function_2 = """def function2():
    x = 1
    y = 2
    result = x + y

    print(f"The result is {result}")"""

        class_example = """
class Example:
    def method1(self):
        print("Another method")
        print("With multiple lines")

    def method2(self):
        return "This should be in another chunk" """

        code = function_1 + "\n\n" + function_2 + "\n\n" + class_example
        chunks = self._test_language(chunker, temp_dir, code, "test.py", "python")
        assert chunks[0].content in function_1
        assert chunks[1].content in function_2
        assert chunks[2].content in class_example

    def test_javascript_chunking(self, temp_dir, chunker):
        function_1 = """function processData() {
    const data = fetchData();
    const result = data.map(item => {
        return item.value * 2;
    });
    return result;
}"""

        class_code = """class DataHandler{
    constructor() {
        this.items = [];
    }

    addItem(item) {
        this.items.push(item);
        this.processItem(item);
    }

    processItem(item) {
        console.log(`Processing ${item}`);
        return item.toString();
    }
}"""

        code = function_1 + "\n\n" + class_code
        chunks = self._test_language(chunker, temp_dir, code, "test.js", "javascript")

        assert len(chunks) >= 2
        assert function_1 in chunks[0].content
        assert class_code in chunks[1].content

    def test_typescript_chunking(self, temp_dir, chunker):
        interface = """interface User {
    id: number;
    name: string;
    email: string;
    age: number;
    is_active: boolean;
    created_at: Date;
    updated_at: Date;
    tags: string[];
}"""

        class_start = """
class UserService {
    private users: User[] = [];

    constructor() {
        this.initializeUsers();
    }

    private initializeUsers(): void {
        // Some initialization logic
        this.users = [];
    }"""

        class_methods = """
    public addUser(user: User): void {
        this.users.push(user);
        this.validateUser(user);
    }

    private validateUser(user: User): boolean {
        return user.id > 0 && user.name.length > 0;
    }
}"""

        code = interface + "\n\n" + class_start + class_methods
        chunks = self._test_language(chunker, temp_dir, code, "test.ts", "typescript")

        assert len(chunks) >= 3
        assert chunks[0].content in interface
        assert "class UserService" in class_start
        assert chunks[2].content in class_methods

    def test_rust_chunking(self, temp_dir, chunker):
        struct_def = """#[derive(Debug)]
struct Vehicle {
    make: String,
    model: String,
    year: u32,
}"""

        impl_block = """impl Vehicle{
    fn new(make: String, model: String, year: u32) -> Self {
        Vehicle { make, model, year }
    }

    fn display_info(&self) -> String {
        format!("{} {} ({})", self.make, self.model, self.year)
    }

    fn update_year(&mut self, new_year: u32) {
        self.year = new_year;
        println!("Updated year to {}", new_year);
    }
}"""

        code = struct_def + "\n\n" + impl_block
        chunks = self._test_language(chunker, temp_dir, code, "test.rs", "rust")

        assert struct_def in chunks[0].content or chunks[0].content in struct_def
        assert impl_block in chunks[1].content or chunks[1].content in impl_block

    def test_go_chunking(self, temp_dir, chunker):
        struct_def = """package main

type Person struct {
    Name string
    Age  int
    Gender string
    City   string
    State  string
}"""

        method_def = """
func (p Person) Greet() string {
    return fmt.Sprintf("Hello, %s", p.Name)
}"""

        code = struct_def + "\n" + method_def
        chunks = self._test_language(chunker, temp_dir, code, "test.go", "go")

        assert chunks[0].content.strip() in struct_def or struct_def in chunks[0].content.strip()
        assert chunks[1].content.strip() in method_def or method_def in chunks[1].content.strip()

    def test_java_chunking(self, temp_dir, chunker):
        class_def = """public class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}"""

        chunks = self._test_language(
            chunker, temp_dir, class_def, "Person.java", "java"
        )
        assert chunks[0].content in class_def or class_def in chunks[0].content

    def test_cpp_chunking(self, temp_dir, chunker):
        class_def = """class Person {
private:
    std::string name;
    int age;
public:
    Person(std::string n, int a) : name(n), age(a) {}
};"""

        chunks = self._test_language(chunker, temp_dir, class_def, "test.cpp", "cpp")
        assert chunks[0].content in class_def or class_def in chunks[0].content

    def test_ruby_chunking(self, temp_dir, chunker):
        class_def = """class Person
  attr_accessor :name, :age, :email, :address, :phone

  def initialize(name, age)
    @name = name
    @age = age
    @email = nil
    @address = nil
    @phone = nil
  end

  def update_contact_info(email, address, phone)
    @email = email
    @address = address
    @phone = phone
    validate_contact_info
    send_verification_email
  end

  def validate_contact_info
    raise "Invalid email" unless @email.include?("@")
    raise "Invalid phone" unless @phone.match?(/^\d{10}$/)
    raise "Invalid address" if @address.nil? || @address.empty?
    puts "Contact info validated successfully"
  end

  def send_verification_email
    puts "Sending verification email to #{@email}"
    puts "Please verify your contact information"
    puts "Name: #{@name}"
    puts "Age: #{@age}"
    puts "Address: #{@address}"
    puts "Phone: #{@phone}"
  end
end"""

        chunks = self._test_language(chunker, temp_dir, class_def, "test.rb", "ruby")
        assert len(chunks) >= 2
        assert "class Person" in chunks[0].content
        assert "send_verification_email" in chunks[1].content

    def _test_language(self, chunker, temp_dir, code, filename, language):
        """Helper method to test language chunking"""
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(code)

        # Use chunk_file instead of chunk_code
        chunks = chunker.chunk_file(file_path)

        # Basic assertions
        assert len(chunks) > 0
        assert all(chunk["tag"]["language"] == language for chunk in chunks)

        # Verify chunk sizes
        for chunk in chunks:
            assert len(chunk["content"].strip()) > 0  # No empty chunks
            assert chunk["token_count"] <= chunker.target_tokens + chunker.overlap_token_size

        # Convert dictionary chunks to a simpler format for testing
        return [SimpleChunk(chunk["content"], chunk["token_count"]) for chunk in chunks]

class SimpleChunk:
    """Simple wrapper to maintain compatibility with existing tests"""
    def __init__(self, content, token_count):
        self.content = content
        self.token_count = token_count
