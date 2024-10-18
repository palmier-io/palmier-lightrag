import json

class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def save_dataset(self, qa_pairs):
        with open(self.dataset_path, 'w') as f:
            json.dump(qa_pairs, f)

    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
