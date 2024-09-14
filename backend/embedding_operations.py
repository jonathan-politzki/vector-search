import numpy as np
import requests
from typing import List, Tuple
import logging
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv('API_TOKEN')
API_URL = os.getenv('API_URL')

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_embedding(text: str, max_retries: int = 3, retry_delay: float = 1.0) -> np.ndarray:
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
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    logger.error(f"All API attempts failed for '{text}'.")
    return None

def perform_operation(positive_words: List[str], negative_words: List[str]) -> np.ndarray:
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

    if positive_embeddings or negative_embeddings:
        positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(768)  # all-mpnet-base-v2 uses 768-dim embeddings
        negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(768)
        result_vector = positive_sum - negative_sum
        result_vector /= np.linalg.norm(result_vector)  # Normalize the result vector
        logger.info("Computed and normalized result vector.")
        return result_vector
    else:
        logger.error("No valid embeddings found for the provided words.")
        raise ValueError("No valid embeddings found for the provided words.")

def find_most_similar(result_vector: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
    """Find top_n similar words using Hugging Face API."""
    logger.info("Finding most similar words using Hugging Face API.")
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": {"source_sentence": result_vector.tolist(), "sentences": [""]}}
        )
        response.raise_for_status()
        similar_words = response.json()
        
        # Process and format the results
        formatted_results = []
        for word, score in similar_words[:top_n]:
            formatted_results.append((word, 1 - score))  # Convert similarity score to distance
            logger.info(f"Word: {word}, Distance: {1 - score}")
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error finding similar words: {str(e)}")
        return []