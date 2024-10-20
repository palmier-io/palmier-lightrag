from utils.llm_interface import evaluate_answer
from tqdm import tqdm

class AnswerEvaluator:
    def evaluate(self, results):
        answer_evaluations = []
        # Add a progress bar
        for result in tqdm(results, desc="Evaluating answers", unit="answer"):
            expected_answer = result['expected_answer']
            model_answer = result['model_answer']
            evaluation = evaluate_answer(expected_answer, model_answer)
            answer_evaluations.append(evaluation)
        return self._aggregate_scores(answer_evaluations)

    def _aggregate_scores(self, evaluations):
        if not evaluations:
            return {
                'accuracy': 0,
                'completeness': 0,
                'relevance': 0,
                'clarity': 0,
                'conciseness': 0,
                'weighted_score': 0
            }

        aggregated = {
            'accuracy': 0,
            'completeness': 0,
            'relevance': 0,
            'clarity': 0,
            'conciseness': 0,
            'weighted_score': 0
        }

        for evaluation in evaluations:
            for metric in aggregated.keys():
                aggregated[metric] += evaluation[metric]['score'] if metric != 'weighted_score' else evaluation[metric]

        num_evaluations = len(evaluations)
        for metric in aggregated.keys():
            aggregated[metric] /= num_evaluations

        return aggregated
