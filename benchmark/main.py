import argparse
from dataset_creation.codebase_chunker import CodebaseChunker
from dataset_creation.qa_generator import QAGenerator
from dataset_creation.dataset_manager import DatasetManager
from benchmarking.model_tester import ModelTester
from benchmarking.context_evaluator import ContextEvaluator
from benchmarking.answer_evaluator import AnswerEvaluator

def main():
    parser = argparse.ArgumentParser(description="Codebase Benchmarking System")
    parser.add_argument("--phase", choices=["dataset", "benchmark"], required=True)
    parser.add_argument("--codebase_path", help="Path to the codebase")
    parser.add_argument("--dataset_path", help="Path to the dataset")
    parser.add_argument("--model_path", help="Path to the model to test")
    args = parser.parse_args()

    if args.phase == "dataset":
        chunker = CodebaseChunker(args.codebase_path)
        chunks = chunker.chunk_codebase()
        
        qa_generator = QAGenerator(chunks)
        qa_pairs = qa_generator.generate_qa_pairs()
        
        dataset_manager = DatasetManager(args.dataset_path)
        dataset_manager.save_dataset(qa_pairs)
        
    elif args.phase == "benchmark":
        model_tester = ModelTester(args.model_path)
        dataset_manager = DatasetManager(args.dataset_path)
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
