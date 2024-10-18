from utils.llm_interface import evaluate_answer

class AnswerEvaluator:
    def evaluate(self, results):
        answer_scores = []
        for result in results:
            expected_answer = result['expected_answer']['answer']
            model_answer = result['model_answer']
            score = evaluate_answer(expected_answer, model_answer)
            answer_scores.append(score)
        return self._aggregate_scores(answer_scores)

    def _aggregate_scores(self, scores):
        # Implement aggregation logic here
        return sum(scores) / len(scores) if scores else 0
