# backend/embedding_operations.py

import numpy as np
import faiss
import openai
import json
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load precomputed embeddings
try:
    with open('embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)
    logger.info("Loaded precomputed embeddings successfully.")
except FileNotFoundError:
    logger.error("embeddings.json file not found.")
    embeddings_dict = {}
except json.JSONDecodeError:
    logger.error("embeddings.json is not a valid JSON file.")
    embeddings_dict = {}

# Convert lists to numpy arrays
embeddings_dict = {word: np.array(embedding) for word, embedding in embeddings_dict.items()}

def get_embedding(text, model='text-embedding-3-small'):
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response['data'][0]['embedding']
        logger.info(f"Obtained embedding for '{text}'.")
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error getting embedding for '{text}': {str(e)}")
        return None

def perform_operation(positive_words, negative_words):
    logger.info("Starting perform_operation.")
    
    # Fetch embeddings for all words using explicit conditionals
    positive_embeddings = [
        embeddings_dict[word] if word in embeddings_dict else get_embedding(word) 
        for word in positive_words
    ]
    negative_embeddings = [
        embeddings_dict[word] if word in embeddings_dict else get_embedding(word) 
        for word in negative_words
    ]

    # Remove None values (failed embeddings)
    positive_embeddings = [emb for emb in positive_embeddings if emb is not None]
    negative_embeddings = [emb for emb in negative_embeddings if emb is not None]

    logger.info(f"Positive embeddings count: {len(positive_embeddings)}")
    logger.info(f"Negative embeddings count: {len(negative_embeddings)}")

    # Update embeddings_dict with new embeddings
    for word, emb in zip(positive_words + negative_words, positive_embeddings + negative_embeddings):
        if word not in embeddings_dict and emb is not None:
            embeddings_dict[word] = emb
            logger.info(f"Added new embedding for word: '{word}'")

    # Ensure there is at least one embedding
    if positive_embeddings or negative_embeddings:
        try:
            dimension = len(next(iter(embeddings_dict.values())))
            logger.info(f"Embedding dimension: {dimension}")
        except StopIteration:
            logger.error("embeddings_dict is empty. Cannot determine embedding dimension.")
            raise ValueError("embeddings_dict is empty.")

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
