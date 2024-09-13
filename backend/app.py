# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import (
    perform_operation,
    find_most_similar_faiss,
    build_faiss_index,
    load_corpus_embeddings,
    save_faiss_index,
    load_faiss_index
)
import threading
import logging
import numpy as np
import os
import nltk
from nltk.corpus import words as nltk_words

# Initialize NLTK
nltk.download('words')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Lock for thread safety
lock = threading.Lock()

# Define a manageable corpus using NLTK's word list
# For testing, use a smaller subset
CORPUS = nltk_words.words()[:1000] + [
    'spacecraft', 'rocket', 'satellite', 'astronaut', 'galaxy',
    'telescope', 'orbit', 'launch', 'module', 'alien', 'blue', 'orange',
    'apple', 'banana', 'car', 'dog', 'elephant', 'forest', 'garden',
    'house', 'island', 'jungle', 'kite', 'lion', 'mountain', 'night',
    'ocean', 'pizza', 'queen', 'river', 'sun', 'tree', 'umbrella',
    'village', 'window', 'xylophone', 'yacht', 'zebra'
    # Add more domain-specific words if needed
]

INDEX_FILE = 'faiss.index'

# Load or build FAISS index
if os.path.exists(INDEX_FILE):
    faiss_index = load_faiss_index(INDEX_FILE)
    corpus_embeddings_dict = load_corpus_embeddings(CORPUS)
    words = list(corpus_embeddings_dict.keys())
    logger.info("FAISS index loaded successfully.")
else:
    try:
        corpus_embeddings_dict = load_corpus_embeddings(CORPUS)
        faiss_index, words = build_faiss_index(corpus_embeddings_dict)
        save_faiss_index(faiss_index, INDEX_FILE)
        logger.info("FAISS index built and saved successfully.")
    except Exception as e:
        logger.error("Failed to build FAISS index from the corpus.", exc_info=True)
        raise e

@app.route('/api/operate', methods=['POST'])
def operate():
    data = request.get_json()
    positive = data.get('positive', [])
    negative = data.get('negative', [])

    if not positive and not negative:
        return jsonify({'error': 'No words provided'}), 400

    try:
        with lock:
            logger.info(f"Received operation with positive: {positive}, negative: {negative}")
            result_vector = perform_operation(positive, negative)

            similar_words = find_most_similar_faiss(result_vector, faiss_index, words)
            formatted_results = [
                {'word': word, 'distance': float(distance)}
                for word, distance in similar_words
            ]
            logger.info(f"Operation successful. Returning results: {formatted_results}")
            return jsonify(formatted_results)
    except Exception as e:
        logger.error("Error in /api/operate", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
