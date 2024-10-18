from utils.llm_interface import generate_qa_pair

class QAGenerator:
    def __init__(self, chunks):
        self.chunks = chunks

    def generate_qa_pairs(self):
        qa_pairs = []
        for chunk in self.chunks:
            qa_pair = generate_qa_pair(chunk)
            qa_pairs.append(qa_pair)
        return qa_pairs
