from utils.llm_interface import get_model_response
from tqdm import tqdm

class ModelTester:
    def __init__(self, model_name):
        self.model_name = model_name

    def test_model(self, qa_pairs):
        results = []
        # Create a progress bar
        pbar = tqdm(total=len(qa_pairs), desc="Testing model", unit="pair")
        
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            expected_answer = qa_pair['answer']
            expected_context_file = qa_pair['context']['file']
            expected_context_chunk = qa_pair['context']['chunk_text']
            
            model_response = get_model_response(self.model_name, question)
            model_answer = model_response['answer']
            model_context_file = model_response['context_file']
            model_context_chunk = model_response['context_chunk']
            
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'model_answer': model_answer,
                'expected_context_file': expected_context_file,
                'expected_context_chunk': expected_context_chunk,
                'model_context_file': model_context_file,
                'model_context_chunk': model_context_chunk
            })
            
            # Update the progress bar
            pbar.update(1)
        
        # Close the progress bar
        pbar.close()
        return results
