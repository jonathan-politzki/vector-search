# test_embedding.py

import openai
import numpy as np
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(text, model='text-embedding-3-small'):
    try:
        text = text.replace("\n", " ")
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        # Access embedding using attribute notation
        embedding = response.data[0].embedding
        logger.info(f"Obtained embedding for '{text}'.")
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {str(e)}")
        return None

if __name__ == "__main__":
    words = ['king', 'queen', 'man', 'woman']
    for word in words:
        emb = get_embedding(word)
        if emb is not None:
            print(f"Embedding for '{word}' retrieved successfully.")
        else:
            print(f"Failed to retrieve embedding for '{word}'.")
