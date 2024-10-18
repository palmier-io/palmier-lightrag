from utils.llm_interface import get_model_response

class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path

    def test_model(self, qa_pairs):
        results = []
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            expected_answer = qa_pair['answer']
            model_answer, context = get_model_response(self.model_path, question)
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'model_answer': model_answer,
                'context': context
            })
        return results
