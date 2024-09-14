# embedding_operations.py

import numpy as np
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class EmbeddingManager:
    def __init__(self, model='text-embedding-3-small'):
        self.model = model

    def get_embedding(self, word):
        """Fetch embedding from OpenAI API."""
        try:
            response = client.embeddings.create(
                input=word,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding).astype('float32')
            logger.info(f"Obtained embedding for '{word}'.")
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{word}': {str(e)}")
            return None

    def perform_operation(self, positive_words, negative_words):
        """Compute result vector by adding positive embeddings and subtracting negative embeddings."""
        logger.info("Starting perform_operation.")
        positive_embeddings = []
        negative_embeddings = []

        for word in positive_words:
            embedding = self.get_embedding(word)
            if embedding is not None:
                positive_embeddings.append(embedding)

        for word in negative_words:
            embedding = self.get_embedding(word)
            if embedding is not None:
                negative_embeddings.append(embedding)

        if positive_embeddings or negative_embeddings:
            positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(len(embedding))
            negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(len(embedding))
            result_vector = positive_sum - negative_sum
            # Normalize the result vector
            result_vector = result_vector / np.linalg.norm(result_vector)
            logger.info("Computed result vector.")
            return result_vector
        else:
            logger.error("No valid embeddings found for the provided words.")
            raise ValueError("No valid embeddings found for the provided words.")

    def find_similar(self, result_vector, top_n=5):
        """Find top_n similar words using OpenAI API."""
        try:
            response = client.embeddings.create(
                input=[result_vector.tolist()],
                model=self.model
            )
            similar_words = response.data[0].embedding
            logger.info(f"Found similar words using OpenAI API.")
            return similar_words
        except Exception as e:
            logger.error(f"Error finding similar words: {str(e)}")
            return []

# Instantiate the EmbeddingManager with the desired model
embedding_manager = EmbeddingManager(model='text-embedding-3-small')

def perform_operation(positive_words, negative_words):
    """Perform the vector operation."""
    return embedding_manager.perform_operation(positive_words, negative_words)

def find_most_similar(result_vector, top_n=5):
    """Find most similar words."""
    return embedding_manager.find_similar(result_vector, top_n)