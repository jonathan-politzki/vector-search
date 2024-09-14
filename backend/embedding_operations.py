import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

# Initialize FAISS index with correct dimension
embedding_dimension = model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dimension)
words = []  # Track words added to FAISS

# Lock for thread safety
lock = threading.Lock()

def get_embedding(word):
    """Fetch embedding using Hugging Face's sentence-transformers."""
    try:
        embedding = model.encode(word, convert_to_tensor=False)  # Returns a list
        embedding = np.array(embedding).astype('float32')  # Ensure FAISS compatibility
        embedding /= np.linalg.norm(embedding)  # Normalize the embedding
        logger.info(f"Obtained and normalized embedding for '{word}'.")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{word}': {str(e)}")
        return None

def add_embedding_to_faiss(word, embedding):
    """Add embedding to FAISS index and update words list."""
    with lock:
        if word not in words:
            faiss_index.add(np.expand_dims(embedding, axis=0))
            words.append(word)
            logger.info(f"Added '{word}' to FAISS index.")
        else:
            logger.info(f"'{word}' is already in FAISS index.")

def perform_operation(positive_words, negative_words):
    """Compute result vector by adding positive embeddings and subtracting negative embeddings."""
    logger.info("Starting perform_operation.")
    positive_embeddings = []
    negative_embeddings = []

    # Fetch embeddings for positive words
    for word in positive_words:
        embedding = get_embedding(word)
        if embedding is not None:
            add_embedding_to_faiss(word, embedding)
            positive_embeddings.append(embedding)
        else:
            logger.warning(f"Skipping word '{word}' due to failed embedding retrieval.")

    # Fetch embeddings for negative words
    for word in negative_words:
        embedding = get_embedding(word)
        if embedding is not None:
            add_embedding_to_faiss(word, embedding)
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
            result_vector /= norm
        logger.info("Computed and normalized result vector.")
        return result_vector
    else:
        logger.error("No valid embeddings found for the provided words.")
        raise ValueError("No valid embeddings found for the provided words.")

def find_most_similar_faiss(result_vector, top_n=5):
    """Find top_n similar words using FAISS index."""
    logger.info("Finding most similar words using FAISS.")
    if len(words) == 0:
        logger.error("FAISS index is empty.")
        return []
    
    result_vector = np.expand_dims(result_vector, axis=0).astype('float32')
    distances, indices = faiss_index.search(result_vector, top_n)
    
    similar_words = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(words):
            similar_words.append((words[idx], float(distance)))
            logger.info(f"Word: {words[idx]}, Distance: {distance}")
        else:
            logger.warning(f"Index {idx} out of bounds for words list.")
    
    logger.info(f"Found similar words: {similar_words}")
    return similar_words
