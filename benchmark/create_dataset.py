import yaml
from dataset_creation.codebase_chunker import CodebaseChunker
from dataset_creation.qa_generator import QAGenerator
from dataset_creation.dataset_manager import DatasetManager

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def main():
    # Load configuration
    config = load_config('./benchmark_config.yaml')
    create_dataset_config = config['create_dataset']
    
    # Dataset creation phase
    print("Starting dataset creation phase...")
    chunker = CodebaseChunker(create_dataset_config['codebase_path'])
    chunks = chunker.chunk_codebase()
    
    qa_generator = QAGenerator(chunks)
    qa_pairs = qa_generator.generate_qa_pairs()
    
    dataset_manager = DatasetManager(create_dataset_config['dataset_path'])
    dataset_manager.save_dataset(qa_pairs)
    print(f"Dataset created and saved to {create_dataset_config['dataset_path']}")

if __name__ == "__main__":
    main()
