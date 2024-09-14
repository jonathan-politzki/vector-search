# backend/embedding_operations.py

import numpy as np
import faiss
import openai
import logging
from dotenv import load_dotenv
import os
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class EmbeddingManager:
    def __init__(self, model='text-embedding-3-small'):
        self.model = model
        # Mapping model names to their dimensions
        model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,  # For reference
        }
        self.dimension = model_dimensions.get(self.model)
        if self.dimension is None:
            logger.error(f"Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}")
        
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.words = []
        self.embeddings_cache = {}
        self.lock = threading.Lock()

    def get_embedding(self, word):
        """Fetch embedding from cache or OpenAI API."""
        with self.lock:
            if word in self.embeddings_cache:
                logger.info(f"Using cached embedding for '{word}'.")
                return self.embeddings_cache[word]

        try:
            # Correct API call without instantiating OpenAI
            response = openai.Embedding.create(
                input=word,
                model=self.model
            )
            # Access embedding using attribute notation
            embedding = response.data[0].embedding
            embedding = np.array(embedding).astype('float32')
            logger.info(f"Obtained embedding for '{word}'.")
            with self.lock:
                self.embeddings_cache[word] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for '{word}': {str(e)}")
            return None

    def add_embedding_to_index(self, word, embedding):
        """Add embedding to FAISS index and words list."""
        with self.lock:
            if len(embedding) != self.dimension:
                logger.error(f"Embedding dimension mismatch for '{word}'. Expected {self.dimension}, got {len(embedding)}.")
                return
            self.faiss_index.add(np.expand_dims(embedding, axis=0))
            self.words.append(word)
            logger.info(f"Added '{word}' to FAISS index.")

    def load_corpus_embeddings(self, corpus):
        """Load embeddings for the corpus and add to FAISS index."""
        logger.info("Loading corpus embeddings.")
        for word in corpus:
            embedding = self.get_embedding(word)
            if embedding is not None:
                self.add_embedding_to_index(word, embedding)
        logger.info(f"Loaded embeddings for {len(self.words)} words.")
        return self.faiss_index, self.words

    def perform_operation(self, positive_words, negative_words):
        """Compute result vector by adding positive embeddings and subtracting negative embeddings."""
        logger.info("Starting perform_operation.")
        positive_embeddings = []
        negative_embeddings = []

        for word in positive_words:
            embedding = self.get_embedding(word)
            if embedding is not None:
                self.add_embedding_to_index(word, embedding)
                positive_embeddings.append(embedding)

        for word in negative_words:
            embedding = self.get_embedding(word)
            if embedding is not None:
                self.add_embedding_to_index(word, embedding)
                negative_embeddings.append(embedding)

        if positive_embeddings or negative_embeddings:
            positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(self.dimension)
            negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(self.dimension)
            result_vector = positive_sum - negative_sum
            logger.info("Computed result vector.")
            return result_vector
        else:
            logger.error("No valid embeddings found for the provided words.")
            raise ValueError("No valid embeddings found for the provided words.")

    def find_similar(self, result_vector, top_n=5):
        """Find top_n similar words using FAISS index."""
        with self.lock:
            if len(self.words) == 0:
                logger.error("FAISS index is empty.")
                return []

            result_vector = np.array([result_vector]).astype('float32')
            distances, indices = self.faiss_index.search(result_vector, top_n)
            similar = [(self.words[i], float(distances[0][idx])) for idx, i in enumerate(indices[0])]
            logger.info(f"Found similar words: {similar}")
            return similar

# Instantiate the EmbeddingManager with the desired model
embedding_manager = EmbeddingManager(model='text-embedding-3-small')  # Change to 'text-embedding-3-large' if needed

def load_corpus_embeddings(corpus):
    """Load corpus embeddings into FAISS index."""
    return embedding_manager.load_corpus_embeddings(corpus)

def perform_operation(positive_words, negative_words):
    """Perform the vector operation."""
    return embedding_manager.perform_operation(positive_words, negative_words)

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    """Find most similar words."""
    return embedding_manager.find_similar(result_vector, top_n)
