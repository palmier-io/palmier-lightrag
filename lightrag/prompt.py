GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = {
    "core_code": [
        "function",
        "class",
        "method",
        "variable",
        "module",
        "package",
        "library",
        "constant",
        "interface"
    ],
    "script": [
        "file",
        "script",
        "command",
        "configuration",
        "dependency",
        "service"
    ],
    "test": [
        "file"
    ],
    "example": [
        "file"
    ],
    "documentation": [
        "file",
        "component",
        "service",
        "architecture",
        "design pattern"
    ],
}



PROMPTS["entity_extraction"] = """
-Goal-
You are given:
2. A file summary providing high-level context of the file.
3. A list of entity types to look for.
4. The actual content (chunk of code or text).

Your objective is to identify:
• Key entities (ignoring subtle/minor variables or implementation details)
• The relationships among those entities, including cross-file relationships
• High-level content keywords capturing the main concepts

-Steps-
0. Determine file type:
   - Infer the file context/type from the `File_summary` by looking for key indicators:
     - Mentions of "deployment steps," "container configuration," or references to infrastructure (e.g., Docker, Kubernetes, AWS) may indicate a deployment script.
     - Mentions of "test fixtures," "mock data," or "example usage" may suggest example/test code.
     - References to "documentation," "overview," "requirements," or "design" may suggest text/documentation.
     - If references to "installation steps," "dependencies," or "environment variables" appear, it may be a setup script.
     - Otherwise, default to core code logic.
   - Do not include the summary text or disclaimers in your final output.

1. Identify significant entities in the file that match the determined file type. For each entity:
   - entity_name: The literal name of the entity as it appears in the code or text.
   - entity_type: Must be one of the entity types specified for the determined file type: [{entity_types}].
   - entity_description: A comprehensive description including:
     • What this entity is and its primary purpose
     • Which file(s) contain this entity
     • How this entity relates to the overall purpose described in the file summary
   - Focus only on important entities - skip minor variables, utility functions, or implementation details
   - Skip any entities that don't match the allowed entity types for this file type.

2. Identify relationships among the recognized entities. For each pair of related entities:
- source_entity: The name of the source entity.
- target_entity: The name of the target entity.
- relationship_description: Must include:
  • How these entities interact or depend on each other
  • Whether they exist in the same file or across different files
  • The nature of their relationship (e.g., inheritance, composition, usage)
- relationship_strength: A numeric score (1–10) indicating the strength or importance of this relationship.
- relationship_keywords: High-level keywords describing the type of relationship.
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Content-Level Keywords:
Identify high-level keywords or topics that describe the overarching themes, concepts, or functionalities of the file. 
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use {record_delimiter} as the list delimiter.

5. Important:
- Avoid hallucinating. Only identify entities and relationships that are explicitly present in the file. If something is implied but not mentioned, do not fabricate it.
- The provided file summary/context is only there to guide you on which details are important based on the inferred or provided file type.

6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
File_summary: {file_summary}
Content: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [function, class, method, variable, module, package, library, constant, interface]
File_summary: This file implements basic mathematical operations through a Calculator class. It's part of the core math module that provides fundamental arithmetic functionality for the application.
Text:
```python
# math_operations.py

class Calculator:

    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y
    else:
        raise ValueError("Cannot divide by zero.")

PI = 3.14159
```
################
Output:
("entity"{tuple_delimiter}"Calculator"{tuple_delimiter}"class"{tuple_delimiter}"Calculator is the main class providing basic arithmetic operations. Located in math_operations.py, it serves as a core component of the math module, implementing the fundamental calculation capabilities described in the file summary."){record_delimiter}
("entity"{tuple_delimiter}"multiply"{tuple_delimiter}"function"{tuple_delimiter}"multiply is a standalone function in math_operations.py that handles multiplication operations. As part of the core math functionality, it complements the Calculator class's arithmetic capabilities."){record_delimiter}
("entity"{tuple_delimiter}"divide"{tuple_delimiter}"function"{tuple_delimiter}"divide is a standalone function in math_operations.py that performs division with error handling. It supports the file's purpose of providing comprehensive mathematical operations with proper validation."){record_delimiter}
("entity"{tuple_delimiter}"PI"{tuple_delimiter}"constant"{tuple_delimiter}"PI is a mathematical constant defined in math_operations.py. While not directly related to the Calculator operations, it provides essential mathematical constants for calculations."){record_delimiter}
("relationship"{tuple_delimiter}"Calculator"{tuple_delimiter}"add"{tuple_delimiter}"The add method is defined within the Calculator class in math_operations.py. It's an internal class method relationship providing basic addition functionality."{tuple_delimiter}"class-method relationship, same file"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"divide"{tuple_delimiter}"ValueError"{tuple_delimiter}"The divide function in math_operations.py raises a ValueError for division by zero, demonstrating error handling integration within the same file."{tuple_delimiter}"error handling, exception, same file"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"arithmetic operations, calculator implementation, mathematical functions, error handling"){completion_delimiter}
######################""",
    """Example 2:

Entity_types: [function, class, method, variable, module, package, library, constant, interface, component, system, process, requirement, specification, architecture, design pattern]
File_summary: This authentication system documentation outlines the core authentication architecture. It describes how the system handles user authentication, focusing on OAuth2 implementation and secure token management.
Text:
# Project Overview Document

The Authentication Module is a critical component of our system architecture. It handles user login, registration, and session management. The module relies on the UserManager class, which interfaces with the database to retrieve and store user information.

In the latest update, we've implemented the OAuth2 authentication process to enhance security. The AuthService class utilizes the TokenGenerator function to create secure tokens.

Below is a snippet of the AuthService class:

```python
class AuthService:
    def __init__(self, user_manager):
        self.user_manager = user_manager

    def authenticate(self, credentials):
        user = self.user_manager.get_user(credentials.username)
        if user and user.verify_password(credentials.password):
            return TokenGenerator.generate_token(user)
        else:
            raise AuthenticationError("Invalid credentials.")

class UserManager:
    def get_user(self, username):
        # Logic to retrieve a user from the database
        pass
```
#############
Output:
("entity"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"component"{tuple_delimiter}"The Authentication Module is the core authentication component described in the documentation. Located in the system's authentication layer, it directly implements the authentication architecture outlined in the file summary, handling user login, registration, and session management."){record_delimiter}
("entity"{tuple_delimiter}"AuthService"{tuple_delimiter}"class"{tuple_delimiter}"AuthService is the primary authentication handling class. Found in the authentication implementation file, it fulfills the secure token management requirements specified in the file summary through OAuth2 integration."){record_delimiter}
("entity"{tuple_delimiter}"UserManager"{tuple_delimiter}"class"{tuple_delimiter}"UserManager is the database interface class for user operations. While in the same file as AuthService, it handles the user data persistence layer mentioned in the authentication architecture documentation."){record_delimiter}
("entity"{tuple_delimiter}"OAuth2"{tuple_delimiter}"process"{tuple_delimiter}"OAuth2 is the authentication protocol implemented in the Authentication Module. As highlighted in the file summary, it's a key security enhancement in the latest system update."){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"UserManager"{tuple_delimiter}"AuthService depends on UserManager for user data retrieval within the same authentication system. They exist in the same file and interact through direct class composition."{tuple_delimiter}"dependency injection, same file"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"OAuth2"{tuple_delimiter}"The Authentication Module implements OAuth2 as its security protocol. This cross-component relationship spans multiple files in the authentication system."{tuple_delimiter}"implementation, security, cross-file"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"TokenGenerator"{tuple_delimiter}"AuthService uses TokenGenerator for creating secure tokens. While TokenGenerator is defined in a separate file, it's integrated into the authentication flow."{tuple_delimiter}"utility usage, cross-file"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"authentication system, OAuth2 implementation, user management, security architecture, token-based authentication"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---
a world class Programming AI assistant designed to help users understand the repository {repository_name}.
Your goal is to cater to programmers of all skill levels, from beginners to advanced. Follow these guidelines to ensure your examples are effective and easy to understand:

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
The data table includes:
1. Summaries of relevant folders and files, including the file path and relevant scores.
2. Relevant entities and relationships with descriptions
3. Relevant code snippets with file path and line numbers

If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with analyzing queries and extracting various types of keywords and search parameters.

---Goal---

Given the query and top k relevant document summaries, analyze the query to extract multiple types of information to help with code search and understanding.

---Instructions---

First, let's break down the analysis process:
1. Understand the main intent of the query
2. Identify the technical domain or context
3. Look for specific technical components mentioned
4. Review the provided document summaries to find:
   - Related components or functionality
   - Similar patterns or implementations
   - Supporting evidence for the query's context
5. Consider related files and components that might be relevant
6. Think about how to expand or refine the query for better search results

Then, analyze the query through these steps and output the analysis in JSON format with the following keys:
- "thought_process": List of reasoning steps that led to your analysis, including insights from document summaries and how you got the answers
- "high_level_keywords": List of overarching concepts, technical patterns, or architectural themes
- "low_level_keywords": List of specific functions, classes, variables, and technical details
- "file_paths": List of potential file paths to search for. Only attempt to include file paths that are most relevant to the query, based on the document summaries. Return empty list if no relevant file paths are found.
- "symbol_names": List of specific code symbols like function names, class names, or variable names that are relevant to the query
- "refined_queries": List of semantic search queries that break down the original query into different aspects or perspectives. Each query should focus on a distinct aspect like:
  - Implementation details
  - Configuration/setup
  - Usage examples
  - Error handling
  - Dependencies/integrations
  - Architecture/design patterns
  - Testing/validation
Make sure each refined query is relevant to answering the original question but explores a different angle.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
Document summaries:
{summary_context}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does the AuthService handle user authentication in the login process?"
Document summaries:
- auth/service.py: Implements the AuthService class with user authentication and token management.
- models/user.py: Contains User model with password hashing and verification methods.
################
Output:
{
  "thought_process": [
    "1. The query is about authentication implementation details",
    "2. It specifically focuses on the AuthService component and login process",
    "3. This likely involves user credentials, token management, and security",
    "4. From summaries: AuthService is implemented in auth/service.py",
    "5. From summaries: User authentication involves password verification from models/user.py",
    "6. Need to look for both high-level auth flow and specific implementation details"
  ],
  "high_level_keywords": ["Authentication", "Login process", "Security"],
  "low_level_keywords": ["User credentials", "Token generation", "Password verification", "Session management"],
  "file_paths": ["auth/service.py", "models/user.py"],
  "symbol_names": ["AuthService", "authenticate", "login", "verify_password", "TokenGenerator", "User"],
  "refined_queries": [
    "What is the implementation flow of AuthService authentication method",
    "How are user credentials validated and verified in the login process",
    "What security measures and token management are used during authentication",
    "How does error handling work for failed authentication attempts",
    "What external dependencies or integrations are used in the authentication process",
    "How is user session state managed after successful authentication",
    "What configuration options are available for AuthService"
  ]
}
#############################""",
    """Example 2:

Query: "How many model providers are supported?"
################
Output:
{
  "thought_process": [
    "1. Query asks about supported model providers in the framework",
    "2. Need to analyze summaries to identify different model integrations",
    "3. Found multiple providers mentioned across different files",
    "4. Examples directory shows various implementations",
    "5. Looking at llm.py and example files for supported integrations",
    "6. Document summaries reveal integration with multiple cloud and open-source providers"
  ],
  "high_level_keywords": ["Model providers", "Language models", "API integration", "Model deployment"],
  "low_level_keywords": [
    "OpenAI compatible",
    "Ollama",
    "LMDeploy",
    "Azure OpenAI",
    "Amazon Bedrock",
  ],
  "file_paths": [
    "lightrag/llm.py",
  ],
  "symbol_names": [
    "openai_complete_if_cache",
    "ollama_model_complete",
    "lmdeploy_model_complete",
    "embedding_func",
    "llm_model_func"
  ],
  "refined_queries": [
    "What are the built-in model provider integrations available",
    "How to configure different model providers in the system",
    "What are the requirements for each supported model provider",
    "How to implement custom model provider integrations",
    "What are the differences between supported provider implementations",
    "How is model provider authentication handled",
    "What are the limitations or constraints for each provider"
  ]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["global_summary"] = """
You are a helpful assistant that creates a global overview of a codebase.
Below are a README, a tree structure of the codebase and a series of summaries extracted from different parts of the codebase (directories, files, components).
Use these summaries to produce a single, coherent, high-level overview that describes:
- The main purpose and functionality of the entire project
- The key components and their roles
- Any notable technologies, patterns, frameworks, or architectures used

Avoid repeating the same information verbatim. Instead, synthesize these points into a concise, readable narrative.

README:
{readme_content}

File Structure:
{tree}

Folder Summaries:
{joined_summaries}

Now, provide a global summary of the entire codebase.
"""

PROMPTS["folder_summary"] = """
You are summarizing code structure, given the codebase structure, directory path, and the summaries of its children.
Summarize its purpose and content concisely and cohesively.
Try your best to explain the directory's purppose and role in the codebase. Return the summary in a single paragraph.

Example:
The directory /configs contains configuration files for a data processing application that focuses on document chunking and summarization using large language models (LLMs), particularly those from OpenAI. The configurations define operational modes (test, development, production), specify the working directory for data storage, and set logging levels. Key features include parameters for chunking text into specified token counts, options for LLM summary generation, and various storage solutions for documents, chunks, vectors, and graphs, supporting backends like JsonKVStorage, Supabase, S3, Qdrant, and Neo4J. Additionally, the configurations address API authentication, optional features such as Stripe metering, and the use of environment variables for sensitive information like API keys, indicating a comprehensive setup for managing and deploying the application effectively."

Codebase structure:
{tree}

Directory path:
{path}

Children summaries:
{children_summaries}

Provide a cohesive summary of the directory's purpose in a single paragraph.
"""

PROMPTS["file_summary"] = """
You are summarizing code structure, given the code content, file path, and the codebase structure.
Summarize its purpose and content concisely and capture the important classes/functions/dependencies of the file.
Try your best to explain the file's purppose and role in the codebase.

Example:
The lightrag/llm.py file defines a QdrantStorage class, which is a vector storage implementation for the Qdrant vector database with multi-tenancy support. This dataclass extends BaseVectorStorage and provides comprehensive methods for interacting with Qdrant, including upsert (inserting or updating vectors), query, query_by_id, and delete operations. Key features include repository-specific filtering, deterministic ID generation, batch processing for embeddings, and robust error handling. The implementation supports vector storage with cosine similarity, handles environment-specific collection naming, and ensures data isolation across different repositories through a repository_id mechanism.

Codebase structure:
{tree}

File path:
{path}

File content:
{content}

Provide a summary of the file's purpose in a single paragraph.
"""
