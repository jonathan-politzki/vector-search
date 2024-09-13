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

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Lock for thread safety
lock = threading.Lock()

# Build Faiss index at startup
try:
    faiss_index, words = build_faiss_index(embeddings_dict)
    app.logger.info("FAISS index built successfully.")
except Exception as e:
    app.logger.error("Failed to build FAISS index on startup.", exc_info=True)
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
            # Declare 'faiss_index' and 'words' as global before using them
            global faiss_index, words

            app.logger.info(f"Received operation with positive: {positive}, negative: {negative}")
            result_vector = perform_operation(positive, negative)

            # Rebuild Faiss index if new words are added
            new_words = [word for word in embeddings_dict.keys() if word not in words]
            if new_words:
                app.logger.info(f"New words detected: {new_words}. Rebuilding FAISS index.")
                faiss_index, words = build_faiss_index(embeddings_dict)

            similar_words = find_most_similar_faiss(result_vector, faiss_index, words)
            formatted_results = [
                {'word': word, 'distance': float(distance)}
                for word, distance in similar_words
            ]
            app.logger.info(f"Operation successful. Returning results: {formatted_results}")
            return jsonify(formatted_results)
    except Exception as e:
        app.logger.error("Error in /api/operate", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
