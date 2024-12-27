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
• All relevant entities of the specified types in the file.
• The relationships among those entities.
• High-level content keywords capturing the main concepts in the file.

-Steps-
0. Determine file type:
   - Infer the file context/type from the `File_summary` by looking for key indicators:
     - Mentions of "deployment steps," "container configuration," or references to infrastructure (e.g., Docker, Kubernetes, AWS) may indicate a deployment script.
     - Mentions of "test fixtures," "mock data," or "example usage" may suggest example/test code.
     - References to "documentation," "overview," "requirements," or "design" may suggest text/documentation.
     - If references to "installation steps," "dependencies," or "environment variables" appear, it may be a setup script.
     - Otherwise, default to core code logic.
   - Do not include the summary text or disclaimers in your final output.

1. Identify all entities in the file that match the determined file type. For each entity:
   - entity_name: The literal name of the entity as it appears in the code or text.
   - entity_type: Must be one of the entity types specified for the determined file type: [{entity_types}].
   - entity_description: A concise description of the entity's attributes, functionalities, or role, in the context/type of the file determined above.
   - Skip any entities that don't match the allowed entity types for this file type.

2. Identify relationships among the recognized entities. For each pair of related entities:
- source_entity: The name of the source entity.
- target_entity: The name of the target entity.
- relationship_description: Explanation of how or why these two entities are connected.
- relationship_strength: A numeric score (1–10) indicating the strength or importance of this relationship.
- relationship_keywords: High-level keywords (e.g., "function call", "inheritance", "dependency", "setup step", "configuration reference").
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Content-Level Keywords:
Identify high-level keywords or topics that describe the overarching themes, concepts, or functionalities of the file. 
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

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
("entity"{tuple_delimiter}"Calculator"{tuple_delimiter}"class"{tuple_delimiter}"Calculator is a simple class that provides methods for basic arithmetic operations like addition and subtraction."){record_delimiter}
("entity"{tuple_delimiter}"add"{tuple_delimiter}"method"{tuple_delimiter}"add is a method of the Calculator class that returns the sum of two numbers, x and y."){record_delimiter}
("entity"{tuple_delimiter}"subtract"{tuple_delimiter}"method"{tuple_delimiter}"subtract is a method of the Calculator class that returns the difference between two numbers, x and y."){record_delimiter}
("entity"{tuple_delimiter}"multiply"{tuple_delimiter}"function"{tuple_delimiter}"multiply is a standalone function that returns the product of x and y."){record_delimiter}
("entity"{tuple_delimiter}"divide"{tuple_delimiter}"function"{tuple_delimiter}"divide is a standalone function that returns the quotient of x and y, raising a ValueError if y is zero."){record_delimiter}
("entity"{tuple_delimiter}"PI"{tuple_delimiter}"constant"{tuple_delimiter}"PI is a constant representing the mathematical constant π, approximately 3.14159."){record_delimiter}
("relationship"{tuple_delimiter}"Calculator"{tuple_delimiter}"add"{tuple_delimiter}"The add method is defined within the Calculator class."{tuple_delimiter}"class-method relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Calculator"{tuple_delimiter}"subtract"{tuple_delimiter}"The subtract method is defined within the Calculator class."{tuple_delimiter}"class-method relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"divide"{tuple_delimiter}"ValueError"{tuple_delimiter}"The divide function raises a ValueError when attempting to divide by zero."{tuple_delimiter}"error handling, exception"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"multiply"{tuple_delimiter}"PI"{tuple_delimiter}"The multiply function could use PI for calculations involving circles."{tuple_delimiter}"mathematical operations, constants"{tuple_delimiter}5){record_delimiter}
("content_keywords"{tuple_delimiter}"arithmetic operations, calculator class, functions, constants, error handling"){completion_delimiter}
######################""",
    """Example 2:

Entity_types: [function, class, method, variable, module, package, library, constant, interface, component, system, process, requirement, specification, architecture, design pattern]
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
("entity"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"component"{tuple_delimiter}"The Authentication Module handles user login, registration, and session management in the system architecture."){record_delimiter}
("entity"{tuple_delimiter}"UserManager"{tuple_delimiter}"class"{tuple_delimiter}"UserManager is a class that interfaces with the database to retrieve and store user information."){record_delimiter} ("entity"{tuple_delimiter}"AuthService"{tuple_delimiter}"class"{tuple_delimiter}"AuthService is a class that handles authentication processes and utilizes the TokenGenerator function to create secure tokens."){record_delimiter}
("entity"{tuple_delimiter}"OAuth2"{tuple_delimiter}"authentication process"{tuple_delimiter}"OAuth2 is an authentication protocol implemented to enhance security in the latest update."){record_delimiter} ("entity"{tuple_delimiter}"TokenGenerator"{tuple_delimiter}"function"{tuple_delimiter}"TokenGenerator is a function used by AuthService to generate secure tokens for authenticated users."){record_delimiter}
("entity"{tuple_delimiter}"AuthenticationError"{tuple_delimiter}"exception"{tuple_delimiter}"AuthenticationError is an exception raised when user credentials are invalid during the authentication process."){record_delimiter}
("relationship"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"UserManager"{tuple_delimiter}"Authentication Module relies on the UserManager class to interface with the database for user information."{tuple_delimiter}"dependency, component-class relationship"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"TokenGenerator"{tuple_delimiter}"AuthService utilizes the TokenGenerator function to create secure tokens."{tuple_delimiter}"function call, utilization"{tuple_delimiter}8){record_delimiter} ("relationship"{tuple_delimiter}"AuthService"{tuple_delimiter}"UserManager"{tuple_delimiter}"AuthService depends on UserManager to retrieve user data during authentication."{tuple_delimiter}"dependency, class interaction"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"OAuth2"{tuple_delimiter}"Authentication Module"{tuple_delimiter}"OAuth2 authentication process is implemented in the Authentication Module to enhance security."{tuple_delimiter}"implementation, security enhancement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"authenticate"{tuple_delimiter}"AuthenticationError"{tuple_delimiter}"The authenticate method raises AuthenticationError when credentials are invalid."{tuple_delimiter}"error handling, exception raising"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"authentication, user management, security, OAuth2, system architecture, dependency injection"){completion_delimiter}
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

You are a helpful AI programming assistant responding to questions about the repository {repository_name}.


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

Please evaluate:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1

Please provide a similarity score between 0 and 1, where:
0: Completely unrelated or answer cannot be reused
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
