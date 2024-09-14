# backend/precompute_embeddings.py

import json
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

def get_embedding(word):
    """Fetch embedding using Hugging Face's sentence-transformers."""
    try:
        embedding = model.encode(word, convert_to_tensor=False)  # Returns a list
        logger.info(f"Obtained embedding for '{word}'.")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{word}': {str(e)}")
        return None

def precompute_embeddings(vocabulary, output_file='embeddings.json'):
    embeddings = {}
    for word in vocabulary:
        embedding = get_embedding(word)
        if embedding:
            embeddings[word] = embedding  # Embeddings are lists, JSON serializable
            logger.info(f"Precomputed embedding for '{word}'.")
        else:
            logger.warning(f"Failed to precompute embedding for '{word}'. Skipping.")
    try:
        with open(output_file, 'w') as f:
            json.dump(embeddings, f)
        logger.info(f"Saved precomputed embeddings to '{output_file}'.")
    except Exception as e:
        logger.error(f"Error saving embeddings to '{output_file}': {str(e)}")

if __name__ == '__main__':
    vocabulary = [
        'king', 'queen', 'man', 'woman', 'prince', 'princess',
        'doctor', 'nurse', 'brother', 'sister', 'husband', 'wife',
        'spacecraft', 'rocket', 'satellite', 'astronaut', 'galaxy',
        'telescope', 'orbit', 'launch', 'module', 'alien'  # Added spacecraft and alien
    ]  # Add more words as needed
    precompute_embeddings(vocabulary)
