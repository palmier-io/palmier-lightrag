from utils.metrics_calculator import calculate_context_metrics
from tqdm import tqdm

class ContextEvaluator:
    def evaluate(self, results):
        file_scores = []
        chunk_scores = []
        combined_scores = []
        
        # Create a progress bar
        pbar = tqdm(total=len(results), desc="Evaluating contexts", unit="context")
        
        for result in results:
            expected_context_file = result['expected_context_file']
            expected_context_chunk = result['expected_context_chunk']
            model_context_file = result['model_context_file']
            model_context_chunk = result['model_context_chunk']
            
            file_score = 1 if expected_context_file == model_context_file else 0
            chunk_score = calculate_context_metrics(expected_context_chunk, model_context_chunk)
            
            # Implement 66/33 split here
            combined_score = (file_score * 0.66) + (chunk_score * 0.33)
            
            file_scores.append(file_score)
            chunk_scores.append(chunk_score)
            combined_scores.append(combined_score)
            
            # Update the progress bar
            pbar.update(1)
        
        # Close the progress bar
        pbar.close()
        
        return self._aggregate_scores(file_scores, chunk_scores, combined_scores)

    def _aggregate_scores(self, file_scores, chunk_scores, combined_scores):
        if not combined_scores:
            return {
                'file_scores': [],
                'chunk_scores': [],
                'combined_scores': [],
                'avg_file_score': 0,
                'avg_chunk_score': 0,
                'avg_combined_score': 0
            }
        
        return {
            'file_scores': file_scores,
            'chunk_scores': chunk_scores,
            'combined_scores': combined_scores,
            'avg_file_score': sum(file_scores) / len(file_scores),
            'avg_chunk_score': sum(chunk_scores) / len(chunk_scores),
            'avg_combined_score': sum(combined_scores) / len(combined_scores)
        }
