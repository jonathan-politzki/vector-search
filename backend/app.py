# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import perform_operation, find_most_similar_faiss, load_corpus_embeddings
import threading
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Lock for thread safety
lock = threading.Lock()

# Define a comprehensive corpus of words
CORPUS = [
    'king', 'queen', 'man', 'woman', 'doctor', 'nurse',
    'spacecraft', 'rocket', 'satellite', 'galaxy',
    'telescope', 'orbit', 'launch', 'module', 'alien',
    'blue', 'orange', 'apple', 'banana', 'car', 'dog',
    'elephant', 'forest', 'garden', 'house', 'island',
    'jungle', 'kite', 'lion', 'mountain', 'night', 'ocean',
    'pizza', 'queen', 'river', 'sun', 'tree', 'umbrella',
    'village', 'window', 'xylophone', 'yacht', 'zebra'
    # Add more words as needed to build a robust FAISS index
]

# Load corpus embeddings and build FAISS index at startup
try:
    load_corpus_embeddings(CORPUS)
    logger.info("FAISS index built successfully from the corpus.")
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
            similar_words = find_most_similar_faiss(result_vector, None, None, top_n=5)
            formatted_results = [{'word': word, 'distance': distance} for word, distance in similar_words]
            logger.info(f"Operation successful. Returning results: {formatted_results}")
            return jsonify(formatted_results)
    except Exception as e:
        logger.error(f"Error in /api/operate: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
