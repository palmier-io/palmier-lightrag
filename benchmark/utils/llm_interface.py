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

def get_model_response(model_path, question):
    # Implement model inference
    pass

def evaluate_answer(expected_answer, model_answer):
    # Implement LLM-based answer evaluation
    pass
