# backend/embedding_operations.py

import numpy as np
import faiss
import openai
import os
from dotenv import load_dotenv
import logging
import nltk
from nltk.corpus import words as nltk_words
import json

# Initialize NLTK
nltk.download('words')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the cache file for embeddings
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.json'

def get_embedding(text, model='text-embedding-ada-002'):
    try:
        response = openai.Embeddings.create(input=[text], model=model)  # Corrected method name and input format
        embedding = response['data'][0]['embedding']  # Using dictionary-style access
        logger.info(f"Obtained embedding for '{text}'.")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {str(e)}")
        return None

def load_embeddings_cache():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'r') as f:
            logger.info("Loading embeddings from cache.")
            return json.load(f)
    return {}

def save_embeddings_cache(embeddings_dict):
    with open(EMBEDDINGS_CACHE_FILE, 'w') as f:
        json.dump(embeddings_dict, f)
    logger.info(f"Embeddings cached to '{EMBEDDINGS_CACHE_FILE}'.")

def load_corpus_embeddings(corpus, model='text-embedding-ada-002'):
    logger.info("Starting to load corpus embeddings.")
    embeddings_dict = load_embeddings_cache()

    for word in corpus:
        if word in embeddings_dict:
            logger.info(f"Embedding for '{word}' loaded from cache.")
            continue  # Skip if already cached
        embedding = get_embedding(word, model=model)
        if embedding is not None:
            embeddings_dict[word] = embedding
            logger.info(f"Loaded embedding for '{word}'.")
            save_embeddings_cache(embeddings_dict)  # Save incrementally
        else:
            logger.warning(f"Failed to load embedding for '{word}'. Skipping.")

    logger.info(f"Successfully loaded embeddings for {len(embeddings_dict)} out of {len(corpus)} words.")
    return embeddings_dict

def perform_operation(positive_words, negative_words):
    logger.info("Starting perform_operation.")

    # Fetch embeddings for positive and negative words
    positive_embeddings = [get_embedding(word) for word in positive_words]
    negative_embeddings = [get_embedding(word) for word in negative_words]

    # Remove None values (failed embeddings)
    positive_embeddings = [emb for emb in positive_embeddings if emb is not None]
    negative_embeddings = [emb for emb in negative_embeddings if emb is not None]

    logger.info(f"Positive embeddings count: {len(positive_embeddings)}")
    logger.info(f"Negative embeddings count: {len(negative_embeddings)}")

    # Ensure there is at least one embedding
    if positive_embeddings or negative_embeddings:
        try:
            dimension = len(next(iter(positive_embeddings + negative_embeddings)))
            logger.info(f"Embedding dimension: {dimension}")
        except StopIteration:
            logger.error("No embeddings were successfully fetched.")
            raise ValueError("No valid embeddings found for the provided words.")

        positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(dimension)
        negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(dimension)

        logger.info("Computed positive and negative sums.")

        # Compute result vector
        result_vector = positive_sum - negative_sum
        logger.info("Computed result vector.")
        return result_vector
    else:
        logger.error("No valid embeddings found for the provided words.")
        raise ValueError("No valid embeddings found for the provided words.")

def build_faiss_index(embeddings_dict):
    logger.info("Building FAISS index.")
    words = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values())).astype('float32')
    
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        logger.error("Embeddings array is empty or has incorrect dimensions.")
        raise ValueError("Embeddings array is empty or has incorrect dimensions.")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info(f"FAISS index built with {len(words)} vectors.")
    return index, words

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    logger.info("Finding most similar words using FAISS.")
    result_vector = np.array([result_vector]).astype('float32')
    distances, indices = faiss_index.search(result_vector, top_n)
    similar = [(words[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    logger.info(f"Found similar words: {similar}")
    return similar

# Functions to persist FAISS index
def save_faiss_index(index, filepath='faiss.index'):
    faiss.write_index(index, filepath)
    logger.info(f"FAISS index saved to '{filepath}'.")

def load_faiss_index(filepath='faiss.index'):
    if os.path.exists(filepath):
        index = faiss.read_index(filepath)
        logger.info(f"FAISS index loaded from '{filepath}'.")
        return index
    else:
        logger.warning(f"FAISS index file '{filepath}' does not exist.")
        return None
