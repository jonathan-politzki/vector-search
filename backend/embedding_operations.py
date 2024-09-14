import numpy as np
import requests
from typing import List, Tuple, Optional
import logging
from dotenv import load_dotenv
import os
import time
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_TOKEN = os.getenv('API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"

if not API_TOKEN:
    raise ValueError("API_TOKEN is not set in the .env file")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def get_embedding(text: str, max_retries: int = 5, initial_delay: float = 1.0) -> Optional[np.ndarray]:
    """Fetch embedding using Hugging Face API with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # Corrected payload to send a list of sentences
            payload = {"inputs": [text]}  # Send text wrapped in a list

            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            # Ensure the response is a list (the embedding vector)
            data = response.json()

            if isinstance(data, dict) and "error" in data:
                logger.error(f"API Error: {data['error']}")
                return None

            # The API should return a list with one element (the embedding vector)
            if isinstance(data, list) and len(data) > 0:
                embedding = np.array(data[0])
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logger.warning(f"Zero norm encountered for embedding of '{text}'. Skipping normalization.")
                    return embedding
                embedding /= norm  # Normalize the embedding
                logger.info(f"Obtained and normalized embedding for '{text}'.")
                return embedding
            else:
                logger.error(f"Unexpected API response format for '{text}': {data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying after {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    logger.error(f"All API attempts failed for '{text}'.")
    return None

def perform_operation(positive_words: List[str], negative_words: List[str]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Compute result vector and return individual word embeddings."""
    logger.info("Starting perform_operation.")
    word_embeddings = []

    for word in positive_words + negative_words:
        embedding = get_embedding(word)
        if embedding is not None:
            word_embeddings.append((word, embedding))
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    if not word_embeddings:
        logger.error("No valid embeddings found for the provided words.")
        return np.zeros(768), []  # Return zero vector if no valid embeddings

    result_vector = np.zeros(768)  # all-mpnet-base-v2 uses 768-dim embeddings
    for i, (word, embedding) in enumerate(word_embeddings):
        if i < len(positive_words):
            result_vector += embedding
        else:
            result_vector -= embedding

    norm = np.linalg.norm(result_vector)
    if norm < 1e-8:
        logger.warning("Resulting vector has very small magnitude. Using average of input embeddings.")
        embeddings = [emb for _, emb in word_embeddings]
        if embeddings:
            result_vector = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(result_vector)
            if norm != 0:
                result_vector /= norm  # Normalize the result vector
        else:
            result_vector = np.zeros(768)
    else:
        result_vector /= norm  # Normalize the result vector

    logger.info("Computed result vector.")
    return result_vector, word_embeddings

def find_most_similar(result_vector: np.ndarray, word_embeddings: List[Tuple[str, np.ndarray]], top_n: int = 5) -> List[Tuple[str, float]]:
    """Find top_n most similar words from the provided word embeddings and additional random words."""
    logger.info("Finding most similar words.")
    similarities = []
    for word, embedding in word_embeddings:
        similarity = float(np.dot(result_vector, embedding))
        similarities.append((word, similarity))
    
    # Add some random words to ensure we always have at least top_n results
    random_words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]
    random.shuffle(random_words)
    for word in random_words:
        if word not in [w for w, _ in word_embeddings]:
            embedding = get_embedding(word)
            if embedding is not None:
                similarity = float(np.dot(result_vector, embedding))
                similarities.append((word, similarity))
                if len(similarities) >= top_n:
                    break
    
    if not similarities:
        logger.error("No similarities to process.")
        return []
    
    # Normalize similarities to be between 0 and 1
    try:
        max_similarity = max(sim for _, sim in similarities)
        min_similarity = min(sim for _, sim in similarities)
    except ValueError:
        logger.error("No similarity scores available.")
        return []

    if max_similarity == min_similarity:
        logger.warning("All similarity scores are identical. Assigning a default normalized similarity of 1.0 to all.")
        normalized_similarities = [(word, 1.0) for word, _ in similarities]
    else:
        normalized_similarities = [
            (word, (sim - min_similarity) / (max_similarity - min_similarity))
            for word, sim in similarities
        ]
    
    normalized_similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = normalized_similarities[:top_n]
    
    logger.info(f"Top {top_n} similar words: {top_similar}")
    return top_similar
