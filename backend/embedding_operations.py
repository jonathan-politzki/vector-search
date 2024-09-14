import numpy as np
import requests
from typing import List, Tuple, Optional
import logging
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv('API_TOKEN')
API_URL = os.getenv('API_URL')

if not API_URL:
    raise ValueError("API_URL is not set in the .env file")

if not API_TOKEN:
    raise ValueError("API_TOKEN is not set in the .env file")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_embedding(text: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[np.ndarray]:
    """Fetch embedding using Hugging Face API with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
            response.raise_for_status()
            embedding = np.array(response.json()[0])
            embedding /= np.linalg.norm(embedding)  # Normalize the embedding
            logger.info(f"Obtained and normalized embedding for '{text}'.")
            return embedding
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    logger.error(f"All API attempts failed for '{text}'.")
    return None

def perform_operation(positive_words: List[str], negative_words: List[str]) -> Optional[np.ndarray]:
    """Compute result vector by adding positive embeddings and subtracting negative embeddings."""
    logger.info("Starting perform_operation.")
    positive_embeddings = []
    negative_embeddings = []

    for word in positive_words:
        embedding = get_embedding(word)
        if embedding is not None:
            positive_embeddings.append(embedding)
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    for word in negative_words:
        embedding = get_embedding(word)
        if embedding is not None:
            negative_embeddings.append(embedding)
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    if not positive_embeddings and not negative_embeddings:
        logger.error("No valid embeddings found for the provided words.")
        return None

    embedding_dim = 768  # all-mpnet-base-v2 uses 768-dim embeddings
    positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(embedding_dim)
    negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(embedding_dim)
    result_vector = positive_sum - negative_sum
    norm = np.linalg.norm(result_vector)
    
    if norm == 0:
        logger.error("Resulting vector has zero magnitude.")
        return None
    
    result_vector /= norm  # Normalize the result vector
    logger.info("Computed and normalized result vector.")
    return result_vector

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(result_vector: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
    """Find top_n similar words using cosine similarity with a predefined vocabulary."""
    logger.info("Finding most similar words.")
    try:
        vocabulary = ["king", "queen", "man", "woman", "prince", "princess", "ruler", "monarch", "royal", "crown"]
        
        similarities = []
        for word in vocabulary:
            word_embedding = get_embedding(word)
            if word_embedding is not None:
                similarity = cosine_similarity(result_vector, word_embedding)
                similarities.append((word, similarity))
        
        # Sort by similarity (highest to lowest) and take top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_n]
        
        # Convert similarity to distance (1 - similarity)
        formatted_results = [(word, 1 - score) for word, score in top_similar]
        
        for word, distance in formatted_results:
            logger.info(f"Word: {word}, Distance: {distance:.4f}")
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error finding similar words: {str(e)}")
        return []

if __name__ == "__main__":
    # Example usage
    result = perform_operation(["king"], ["man"])
    if result is not None:
        similar_words = find_most_similar(result)
        print("Similar words:", similar_words)
    else:
        print("Failed to perform operation.")