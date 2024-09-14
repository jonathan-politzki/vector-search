# backend/embedding_operations.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (if any)
load_dotenv()

# Path to the embeddings JSON file
EMBEDDINGS_FILE = 'embeddings.json'

# Load precomputed embeddings
try:
    with open(EMBEDDINGS_FILE, 'r') as f:
        embeddings_dict = json.load(f)
    logger.info("Loaded precomputed embeddings successfully.")
except FileNotFoundError:
    logger.warning(f"{EMBEDDINGS_FILE} not found. Starting with an empty embeddings dictionary.")
    embeddings_dict = {}
except json.JSONDecodeError:
    logger.error(f"{EMBEDDINGS_FILE} is not a valid JSON file. Starting with an empty embeddings dictionary.")
    embeddings_dict = {}

# Convert lists to numpy arrays
embeddings_dict = {word: np.array(embedding) for word, embedding in embeddings_dict.items()}

# Initialize the sentence transformer model
# You can choose a different model from Hugging Face's repository if needed
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

# Determine embedding dimension
if embeddings_dict:
    embedding_dimension = len(next(iter(embeddings_dict.values())))
else:
    embedding_dimension = model.get_sentence_embedding_dimension()
logger.info(f"Using embedding dimension: {embedding_dimension}")

# Initialize FAISS index
if embeddings_dict:
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    words = list(embeddings_dict.keys())
    embeddings_matrix = np.array(list(embeddings_dict.values())).astype('float32')
    faiss_index.add(embeddings_matrix)
    logger.info(f"FAISS index built with {len(words)} vectors.")
else:
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    words = []
    logger.info("Initialized empty FAISS index.")

# Lock for thread safety
lock = threading.Lock()

def save_embeddings():
    """Save the embeddings_dict to the JSON file."""
    with open(EMBEDDINGS_FILE, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_dict = {word: embedding.tolist() for word, embedding in embeddings_dict.items()}
        json.dump(serializable_dict, f)
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}.")

def get_embedding(word):
    """Fetch embedding using Hugging Face's sentence-transformers."""
    if word in embeddings_dict:
        logger.info(f"Using cached embedding for '{word}'.")
        return embeddings_dict[word]
    
    try:
        embedding = model.encode(word)
        embedding = embedding.astype('float32')
        embeddings_dict[word] = embedding
        logger.info(f"Obtained and cached embedding for '{word}'.")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{word}': {str(e)}")
        return None

def perform_operation(positive_words, negative_words):
    """Compute result vector by adding positive embeddings and subtracting negative embeddings."""
    logger.info("Starting perform_operation.")
    positive_embeddings = []
    negative_embeddings = []

    # Fetch embeddings for positive words
    for word in positive_words:
        embedding = get_embedding(word)
        if embedding is not None:
            positive_embeddings.append(embedding)
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    # Fetch embeddings for negative words
    for word in negative_words:
        embedding = get_embedding(word)
        if embedding is not None:
            negative_embeddings.append(embedding)
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    # Perform vector math
    if positive_embeddings or negative_embeddings:
        positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(embedding_dimension, dtype='float32')
        negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(embedding_dimension, dtype='float32')
        result_vector = positive_sum - negative_sum
        # Normalize the result vector
        norm = np.linalg.norm(result_vector)
        if norm != 0:
            result_vector = result_vector / norm
        logger.info("Computed and normalized result vector.")
        
        # Save embeddings after operation
        save_embeddings()
        
        return result_vector
    else:
        logger.error("No valid embeddings found for the provided words.")
        raise ValueError("No valid embeddings found for the provided words.")

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    """Find top_n similar words using FAISS index."""
    logger.info("Finding most similar words using FAISS.")
    if len(words) == 0:
        logger.error("FAISS index is empty.")
        return []
    
    result_vector = np.expand_dims(result_vector, axis=0).astype('float32')
    distances, indices = faiss_index.search(result_vector, top_n)
    similar = [(words[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]
    logger.info(f"Found similar words: {similar}")
    return similar

def build_faiss_index(embeddings_dict):
    """Build FAISS index from the embeddings_dict."""
    logger.info("Building FAISS index.")
    words = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values())).astype('float32')
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    logger.info(f"FAISS index built with {len(words)} vectors.")
    return index, words
