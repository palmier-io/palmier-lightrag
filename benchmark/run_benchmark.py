import yaml
from dataset_creation.dataset_manager import DatasetManager
from benchmarking.model_tester import ModelTester
from benchmarking.context_evaluator import ContextEvaluator
from benchmarking.answer_evaluator import AnswerEvaluator
import json

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def main():
    # Load configuration
    config = load_config('./benchmark_config.yaml')
    run_benchmark_config = config['run_benchmark']
    
    dataset_manager = DatasetManager(run_benchmark_config['dataset_path'])
    # Benchmarking phase
    print("\nStarting benchmarking phase...")
    model_tester = ModelTester(run_benchmark_config['testing_model_name'])
    qa_pairs = dataset_manager.load_dataset()
    
    results = model_tester.test_model(qa_pairs)
    
    context_evaluator = ContextEvaluator()
    context_scores = context_evaluator.evaluate(results)
    
    answer_evaluator = AnswerEvaluator()
    answer_scores = answer_evaluator.evaluate(results)
    
    detailed_results = print_detailed_results(context_scores, answer_scores, run_benchmark_config)
    
    # Save detailed results to a JSON file
    with open(run_benchmark_config['results_path'], 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\nDetailed results saved to {run_benchmark_config['results_path']}")

def calculate_weighted_score(context_scores, answer_scores, config):
    context_weight = config['context_weight']
    answer_weight = config['answer_weight']
    
    context_avg = context_scores['avg_combined_score']
    answer_avg = answer_scores['weighted_score']
    
    weighted_score = (context_avg * context_weight) + (answer_avg * answer_weight)
    return round(weighted_score, 2)

def print_detailed_results(context_scores, answer_scores, config):
    weighted_score = calculate_weighted_score(context_scores, answer_scores, config)
    
    detailed_results = {
        "model_name": config['testing_model_name'],
        "dataset_path": config['dataset_path'],
        "overall_weighted_score": weighted_score,
        "context_evaluation": context_scores,
        "answer_evaluation": answer_scores
    }
    
    print("--------------------------------")
    print(f"Total Evaluation Results for {config['testing_model_name']} on {config['dataset_path']}\n")
    print(f"Overall Weighted Score: {weighted_score} / 1.00")
    print(f"({config['answer_weight']*100}% Answer Evaluation, {config['context_weight']*100}% Context Evaluation)\n")
    
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
