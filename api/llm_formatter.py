from typing import Dict, Any, List
import json
from openai import AsyncOpenAI

# Initialize the OpenAI client
client = AsyncOpenAI()

async def format_response_with_llm(raw_results: List[Dict[str, Any]], schema: Dict[str, Any] = None, additional_params: Dict[str, Any] = None, query: str = None) -> Dict[str, Any]:
    # Construct the prompt
    print(schema)
    prompt = construct_prompt(raw_results, schema, additional_params, query)
    print(prompt)

    # Call the LLM
    response = await call_llm(prompt)

    # Parse and validate the response
    formatted_response = parse_and_validate_response(response, schema)

    return formatted_response

def construct_prompt(raw_results: List[Dict[str, Any]], schema: Dict[str, Any] = None, additional_params: Dict[str, Any] = None, query: str = None) -> str:
    prompt = "Based on the following information, please provide a response:\n\n"
    prompt += "Raw Information:\n"
    prompt += json.dumps(raw_results, indent=2) + "\n\n"
    
    if schema:
        schema_properties = schema['schema']['properties']
        prompt += "Output Schema:\n"
        for prop, details in schema_properties.items():
            prompt += f"- {prop} ({details['type']}): {details['description']}\n"
            prompt += f"  Explanation: {details['x_explanation']}\n"
    elif query:
        prompt += f"Query: {query}\n\n"
        prompt += "Please respond to the query using the raw information provided. Provide a clear and concise answer."
    else:
        prompt += "Please summarize the key information from the raw data provided."
    
    if additional_params:
        prompt += "\nAdditional Parameters:\n"
        prompt += json.dumps(additional_params, indent=2) + "\n"
    
    return prompt

async def call_llm(prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that formats information according to specified schemas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        raise

def parse_and_validate_response(response: str, schema: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        # Attempt to parse the response as JSON
        parsed_response = json.loads(response)
    except json.JSONDecodeError:
        # If parsing fails, attempt to extract a JSON object from the text
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != -1:
            try:
                parsed_response = json.loads(response[start:end])
            except json.JSONDecodeError:
                # If JSON parsing fails, return the response as a string
                return {"response": response.strip()}
        else:
            # If no JSON object is found, return the response as a string
            return {"response": response.strip()}

    if schema:
        # Validate the parsed response against the schema
        schema_properties = schema['schema']['properties']
        required_fields = schema['schema']['required']

        for field in required_fields:
            if field not in parsed_response:
                raise ValueError(f"Required field '{field}' is missing from the LLM response")

        for field, value in parsed_response.items():
            if field in schema_properties:
                expected_type = schema_properties[field]['type']
                if expected_type == 'string' and not isinstance(value, str):
                    parsed_response[field] = str(value)
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    try:
                        parsed_response[field] = float(value)
                    except ValueError:
                        raise ValueError(f"Field '{field}' should be a number")
                elif expected_type == 'integer' and not isinstance(value, int):
                    try:
                        parsed_response[field] = int(value)
                    except ValueError:
                        raise ValueError(f"Field '{field}' should be an integer")
            else:
                # Remove fields that are not in the schema
                del parsed_response[field]
    
    return parsed_response

# Example usage:
# async def main():
#     raw_results = [{"name": "Acme Corp", "sector": "Technology", "employees": 500}]
#     schema = {
#         "schema": {
#             "properties": {
#                 "name": {"type": "string", "description": "Company name", "x_explanation": "Full legal name"},
#                 "industry": {"type": "string", "description": "Industry sector", "x_explanation": "Primary industry"},
#                 "employees": {"type": "integer", "description": "Number of employees", "x_explanation": "Full-time employee count"}
#             },
#             "required": ["name", "industry"]
#         }
#     }
#     formatted_response = await format_response_with_llm(raw_results, schema)
#     print(formatted_response)

# import asyncio
# asyncio.run(main())
