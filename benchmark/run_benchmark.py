from dataset_creation.dataset_manager import DatasetManager
from benchmarking.model_tester import ModelTester
from benchmarking.context_evaluator import ContextEvaluator
from benchmarking.answer_evaluator import AnswerEvaluator

# Hardcoded paths
CODEBASE_PATH = "./"
DATASET_PATH = "./dataset.json"
MODEL_PATH = "./model"

def main():    
    dataset_manager = DatasetManager(DATASET_PATH)
    # Benchmarking phase
    print("\nStarting benchmarking phase...")
    model_tester = ModelTester(MODEL_PATH)
    qa_pairs = dataset_manager.load_dataset()
    
    results = model_tester.test_model(qa_pairs)
    
    context_evaluator = ContextEvaluator()
    context_scores = context_evaluator.evaluate(results)
    
    answer_evaluator = AnswerEvaluator()
    answer_scores = answer_evaluator.evaluate(results)
    
    print("Context Evaluation Scores:", context_scores)
    print("Answer Evaluation Scores:", answer_scores)

if __name__ == "__main__":
    main()