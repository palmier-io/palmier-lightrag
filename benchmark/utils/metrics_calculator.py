from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')

def get_code_embedding(code_snippet, model, tokenizer):
    """
    Converts a code snippet into an embedding using the provided model and tokenizer.
    """
    # Tokenize the code snippet
    inputs = tokenizer(code_snippet, return_tensors='pt', truncation=True, max_length=512)
    # Get the model outputs without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :]
    # Normalize the embedding vector
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

def calculate_context_metrics(expected_context, retrieved_context):
    """
    Calculates the semantic similarity between the expected and retrieved code contexts.

    Args:
        expected_context (str): The expected code context as a string.
        retrieved_context (str): The retrieved code context as a string.

    Returns:
        float: A similarity score between 0 and 1.
    """

    # Check if retrieved_context is empty
    if not retrieved_context.strip():
        return 0.0

    # Get embeddings for both code snippets
    expected_embedding = get_code_embedding(expected_context, model, tokenizer)
    retrieved_embedding = get_code_embedding(retrieved_context, model, tokenizer)

    # Compute cosine similarity
    similarity = torch.cosine_similarity(expected_embedding, retrieved_embedding).item()
    
    # Ensure the similarity score is between 0 and 1
    similarity = max(0.0, min(1.0, (similarity + 1) / 2))
    
    return similarity

def calculate_answer_metrics(expected_answer, model_answer):
    # Implement answer evaluation metrics
    # This is a placeholder and should be replaced with actual implementation
    pass
