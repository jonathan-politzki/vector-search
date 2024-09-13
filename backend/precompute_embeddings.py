# backend/precompute_embeddings.py

import json
import openai
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(text, model='text-embedding-ada-002'):
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response['data'][0]['embedding']
        logger.info(f"Obtained embedding for '{text}'.")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {str(e)}")
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
        'doctor', 'nurse', 'brother', 'sister', 'husband', 'wife'
    ]  # Add more words as needed
    precompute_embeddings(vocabulary)
