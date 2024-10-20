import json
from utils.llm_interface import generate_qa_pair
from tqdm import tqdm

class QAGenerator:
    def __init__(self, chunks, output_file='qa_pairs.json'):
        self.chunks = chunks
        self.output_file = output_file

    def generate_qa_pairs(self):
        qa_pairs = []
        for i, chunk in enumerate(tqdm(self.chunks, desc="Generating Q&A pairs", unit="chunk")):
            qa_pair = generate_qa_pair(chunk, i)
            qa_pairs.append(qa_pair)
            
            # Write to JSON file after each pair is generated
            self._write_to_json(qa_pairs)
            
        return qa_pairs

    def _write_to_json(self, qa_pairs):
        with open(self.output_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
