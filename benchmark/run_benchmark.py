from dataset_creation.dataset_manager import DatasetManager
from benchmarking.model_tester import ModelTester
from benchmarking.context_evaluator import ContextEvaluator
from benchmarking.answer_evaluator import AnswerEvaluator
import json

# Hardcoded paths
DATASET_PATH = "./qa_pairs.json"
MODEL_PATH = "./model"
MODEL_NAME = "gpt-4o-mini"
RESULTS_PATH = "./benchmark_results.json"

def main():    
    dataset_manager = DatasetManager(DATASET_PATH)
    # Benchmarking phase
    print("\nStarting benchmarking phase...")
    model_tester = ModelTester(MODEL_NAME)
    qa_pairs = dataset_manager.load_dataset()
    
    results = model_tester.test_model(qa_pairs)
    
    context_evaluator = ContextEvaluator()
    context_scores = context_evaluator.evaluate(results)
    
    answer_evaluator = AnswerEvaluator()
    answer_scores = answer_evaluator.evaluate(results)
    
    detailed_results = print_detailed_results(context_scores, answer_scores)
    
    # Save detailed results to a JSON file
    with open(RESULTS_PATH, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

def calculate_weighted_score(context_scores, answer_scores):
    context_weight = 0.3
    answer_weight = 0.7
    
    context_avg = context_scores['avg_combined_score']
    answer_avg = answer_scores['weighted_score']
    
    weighted_score = (context_avg * context_weight) + (answer_avg * answer_weight)
    return round(weighted_score, 2)

def print_detailed_results(context_scores, answer_scores):
    weighted_score = calculate_weighted_score(context_scores, answer_scores)
    
    detailed_results = {
        "model_name": MODEL_NAME,
        "dataset_path": DATASET_PATH,
        "overall_weighted_score": weighted_score,
        "context_evaluation": context_scores,
        "answer_evaluation": answer_scores
    }
    
    print("--------------------------------")
    print(f"Total Evaluation Results for {MODEL_NAME} on {DATASET_PATH}\n")
    print(f"Overall Weighted Score: {weighted_score} / 1.00")
    print("(70% Answer Evaluation, 30% Context Evaluation)\n")
    
    print("Context Evaluation Breakdown:")
    print(f"  Average File Score:     {context_scores['avg_file_score']:.2f}")
    print(f"  Average Chunk Score:    {context_scores['avg_chunk_score']:.2f}")
    print(f"  Average Combined Score: {context_scores['avg_combined_score']:.2f}\n")
    
    print("Answer Evaluation Breakdown:")
    print(f"  Accuracy:      {answer_scores['accuracy']:.2f}")
    print(f"  Completeness:  {answer_scores['completeness']:.2f}")
    print(f"  Relevance:     {answer_scores['relevance']:.2f}")
    print(f"  Clarity:       {answer_scores['clarity']:.2f}")
    print(f"  Conciseness:   {answer_scores['conciseness']:.2f}")
    print(f"  Weighted Score: {answer_scores['weighted_score']:.2f}")
    
    return detailed_results

if __name__ == "__main__":
    main()
