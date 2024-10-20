import os
import json
from openai import OpenAI

# Initialize OpenAI client (make sure to set your API key in environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_question_templates():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'questions_template.json')
    with open(file_path, 'r') as f:
        templates = json.load(f)
    return templates['questionTemplates']

def load_prompt_template():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'qa_prompt.txt')
    with open(file_path, 'r') as f:
        return f.read()

def generate_qa_pair(chunk, chunk_index):
    templates = load_question_templates()
    prompt_template = load_prompt_template()

    # Handle both single chunk (dict) and multiple chunks (list of dicts)
    if isinstance(chunk, list):
        # Combine content from multiple chunks
        chunk_content = "\n\n".join(c['content'] for c in chunk)
        chunk_file = chunk[0]['metadata']['file']
    else:
        chunk_content = chunk['content']
        chunk_file = chunk['metadata']['file']

    prompt = prompt_template.format(
        code_chunk=chunk_content,
        templates=json.dumps(templates, indent=2)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Updated model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates relevant question-answer pairs for code chunks."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    try:
        content = response.choices[0].message.content.strip()
        # Extract JSON from the content (remove markdown formatting)
        json_content = content.split('```json\n')[1].split('\n```')[0]
        qa_pair = json.loads(json_content)
        qa_pair['context'] = {
            'file': chunk_file,
            'chunk_text': chunk_content
        }
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        qa_pair = {
            "question": "Error generating question",
            "answer": "Error generating answer",
            "context": {
                "file": chunk_file,
                "chunk_text": chunk_content
            }
        }

    return qa_pair


def get_model_response(model_name, question):
    # Add suppport for other models (i.e. Cursor, and our own RAG implementation on the specific codebase it's beign tested on)
    if model_name == "gpt-4o-mini":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Your task is to answer a codebase related question. Provide your answer in a JSON format with 'answer', 'context_file', and 'context_chunk' fields. If you have no context to answer the question, leave the context fields blank."},
                {"role": "user", "content": question}
            ],
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        # Parse the response content as JSON
        try:
            content = response.choices[0].message.content.strip()
            response_json = json.loads(content)
        except json.JSONDecodeError:
            # If the response is not valid JSON, create a default structure
            response_json = {
                "answer": content,
                "context_file": "",
                "context_chunk": ""
            }
        
        # Ensure the response has the required fields
        if "answer" not in response_json:
            response_json["answer"] = "Error: No answer provided"
        if "context_file" not in response_json:
            response_json["context_file"] = ""
        if "context_chunk" not in response_json:
            response_json["context_chunk"] = ""
        
        return response_json
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def evaluate_answer(expected_answer, model_answer):
    prompt = f"""
    You are an expert / harsh evaluator tasked with assessing the quality of an AI-generated answer compared to an expected answer. Please evaluate the model's answer based on the following criteria:

    1. Accuracy (45%): How factually correct and true to the expected answer is the model's response?
    2. Completeness (25%): How thoroughly does the answer cover all aspects of the expected answer?
    3. Relevance (10%): How well does the answer address the core of the question or topic?
    4. Clarity (10%): How clear and easy to understand is the answer?
    5. Conciseness (10%): How concise and to-the-point is the answer without unnecessary information?

    Expected Answer:
    {expected_answer}

    Model's Answer:
    {model_answer}

    Please provide a score for each criterion (between 0 and 1) and a brief explanation of your reasoning. Output your evaluation in JSON format with the following structure:
    {{
        "accuracy": {{
            "score": <score>,
            "reasoning": "<explanation>"
        }},
        "completeness": {{
            "score": <score>,
            "reasoning": "<explanation>"
        }},
        "relevance": {{
            "score": <score>,
            "reasoning": "<explanation>"
        }},
        "clarity": {{
            "score": <score>,
            "reasoning": "<explanation>"
        }},
        "conciseness": {{
            "score": <score>,
            "reasoning": "<explanation>"
        }}
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert evaluator of AI-generated answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        n=1,
        stop=None,
        temperature=0.5,
    )

    try:
        content = response.choices[0].message.content.strip()
        # Remove any markdown formatting
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        evaluation = json.loads(content)
        
        # Calculate weighted average score
        weighted_score = (
            evaluation['accuracy']['score'] * 0.45 +
            evaluation['completeness']['score'] * 0.25 +
            evaluation['relevance']['score'] * 0.10 +
            evaluation['clarity']['score'] * 0.10 +
            evaluation['conciseness']['score'] * 0.10
        )
        
        evaluation['weighted_score'] = round(weighted_score, 2)
        
        return evaluation
    except json.JSONDecodeError as e:
        print(f"Error parsing evaluation result: {e}")
        print(f"Raw response: {content}")
        return {"error": "Failed to parse evaluation result", "raw_content": content}
    except KeyError as e:
        print(f"Missing key in evaluation result: {e}")
        print(f"Evaluation structure: {evaluation}")
        return {"error": f"Missing key in evaluation result: {e}", "evaluation": evaluation}
