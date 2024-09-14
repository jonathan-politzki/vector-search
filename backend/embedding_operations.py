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

# Initialize the Hugging Face SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace this with the appropriate model

# Initialize FAISS index and embedding dimension
embedding_dimension = model.get_sentence_embedding_dimension()  # Automatically get dimension
faiss_index = faiss.IndexFlatL2(embedding_dimension)
words = []
embeddings_dict = {}

# Lock for thread safety
lock = threading.Lock()

def get_embedding(word):
    """Fetch embedding using Hugging Face's sentence-transformers."""
    if word in embeddings_dict:
        logger.info(f"Using cached embedding for '{word}'.")
        return embeddings_dict[word]
    
    try:
        embedding = model.encode(word, convert_to_tensor=True).cpu().numpy()  # Convert to numpy array
        embedding = embedding.astype('float32')  # Ensure FAISS compatibility
        embeddings_dict[word] = embedding  # Cache the embedding
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
        
        return result_vector
    else:
        logger.error("No valid embeddings found for the provided words.")
        raise ValueError("No valid embeddings found for the provided words.")

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    """Find top_n similar words using FAISS index."""
    logger.info("Finding most similar words using FAISS.")
    
    result_vector = np.expand_dims(result_vector, axis=0).astype('float32')
    
    # Search for nearest neighbors in FAISS
    distances, indices = faiss_index.search(result_vector, top_n)
    
    similar_words = [(words[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]
    logger.info(f"Found similar words: {similar_words}")
    return similar_words

def add_word_to_faiss(word):
    """Add a word to the FAISS index if it's not already there."""
    if word not in words:
        embedding = get_embedding(word)
        if embedding is not None:
            faiss_index.add(np.expand_dims(embedding, axis=0))  # Add embedding to FAISS
            words.append(word)
            logger.info(f"Added '{word}' to FAISS index.")

def build_faiss_index(embeddings_dict):
    """Build FAISS index from the embeddings_dict."""
    logger.info("Building FAISS index.")
    for word in embeddings_dict:
        add_word_to_faiss(word)
    logger.info(f"FAISS index built with {len(words)} vectors.")
    return faiss_index, words
