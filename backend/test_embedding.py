# test_embedding.py

from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text, model='text-embedding-3-small'):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        logger.info(f"Obtained embedding for '{text}'.")
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {str(e)}")
        return None

def test_embeddings():
    words = ['king', 'queen', 'man', 'woman']
    for word in words:
        emb = get_embedding(word)
        if emb is not None:
            print(f"Embedding for '{word}' retrieved successfully. Shape: {emb.shape}")
        else:
            print(f"Failed to retrieve embedding for '{word}'.")

if __name__ == "__main__":
    test_embeddings()