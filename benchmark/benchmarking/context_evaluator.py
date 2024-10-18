from utils.metrics_calculator import calculate_context_metrics

class ContextEvaluator:
    def evaluate(self, results):
        context_scores = []
        for result in results:
            expected_context = result['expected_answer']['context']
            retrieved_context = result['context']
            score = calculate_context_metrics(expected_context, retrieved_context)
            context_scores.append(score)
        return self._aggregate_scores(context_scores)

    def _aggregate_scores(self, scores):
        # Implement aggregation logic here
        return sum(scores) / len(scores) if scores else 0
