from dataset_creation.codebase_chunker import CodebaseChunker
from dataset_creation.qa_generator import QAGenerator
from dataset_creation.dataset_manager import DatasetManager

# Hardcoded paths
CODEBASE_PATH = "./"
DATASET_PATH = "./dataset.json"
MODEL_PATH = "./model"

def main():
    # Dataset creation phase
    print("Starting dataset creation phase...")
    chunker = CodebaseChunker(CODEBASE_PATH)
    chunks = chunker.chunk_codebase()
    
    qa_generator = QAGenerator(chunks)
    qa_pairs = qa_generator.generate_qa_pairs()
    
    dataset_manager = DatasetManager(DATASET_PATH)
    dataset_manager.save_dataset(qa_pairs)
    print(f"Dataset created and saved to {DATASET_PATH}")

if __name__ == "__main__":
    main()
