# backend/app.py

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from embedding_operations import (
    perform_operation,
    find_most_similar_faiss,
    build_faiss_index,
    embeddings_dict
)
import threading

app = Flask(__name__)

# Configure CORS to allow requests from localhost:3000
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Lock for thread safety
lock = threading.Lock()

# Build Faiss index at startup
faiss_index, words = build_faiss_index(embeddings_dict)

@app.route('/api/operate', methods=['POST', 'OPTIONS'])
def operate():
    if request.method == 'OPTIONS':
        # Create a blank response for preflight
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    data = request.get_json()
    positive = data.get('positive', [])
    negative = data.get('negative', [])

    if not positive and not negative:
        return jsonify({'error': 'No words provided'}), 400

    try:
        with lock:
            result_vector = perform_operation(positive, negative)

            # Check if embeddings_dict has been updated
            if any(word not in words for word in embeddings_dict.keys()):
                global faiss_index, words
                faiss_index, words = build_faiss_index(embeddings_dict)

            similar_words = find_most_similar_faiss(result_vector, faiss_index, words)
            # Format results
            formatted_results = [
                {'word': word, 'distance': float(distance)}
                for word, distance in similar_words
            ]
            return jsonify(formatted_results)
    except Exception as e:
        return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
