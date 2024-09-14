# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import (
    perform_operation,
    find_most_similar_faiss,
    build_faiss_index,
    embeddings_dict
)
import threading
import logging
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Lock for thread safety
lock = threading.Lock()

# Build FAISS index at startup
try:
    faiss_index, words = build_faiss_index(embeddings_dict)
    logger.info("FAISS index built successfully.")
except Exception as e:
    logger.error("Failed to build FAISS index on startup.", exc_info=True)
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

            # Identify new words that were added during perform_operation
            new_words = [word for word in embeddings_dict.keys() if word not in words]
            if new_words:
                logger.info(f"New words detected: {new_words}. Adding to FAISS index.")
                for word in new_words:
                    embedding = embeddings_dict[word].astype('float32')
                    faiss_index.add(np.expand_dims(embedding, axis=0))
                    words.append(word)
                logger.info(f"Added {len(new_words)} new words to FAISS index.")

            similar_words = find_most_similar_faiss(result_vector, faiss_index, words, top_n=5)
            formatted_results = [
                {'word': word, 'distance': distance}
                for word, distance in similar_words
            ]
            logger.info(f"Operation successful. Returning results: {formatted_results}")
            return jsonify(formatted_results)
    except Exception as e:
        logger.error("Error in /api/operate", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Perform a simple operation to check if the system is working
        test_vector = perform_operation(['test'], [])
        if test_vector is not None:
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'Failed to perform test operation'}), 500
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
