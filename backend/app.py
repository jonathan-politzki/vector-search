# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import perform_operation, find_most_similar_faiss, build_faiss_index, embeddings_dict
import threading

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Lock for thread safety
lock = threading.Lock()

# Build Faiss index at startup
faiss_index, words = build_faiss_index(embeddings_dict)

@app.route('/api/operate', methods=['POST'])
def operate():
    data = request.get_json()
    positive = data.get('positive', [])
    negative = data.get('negative', [])

    if not positive and not negative:
        return jsonify({'error': 'No words provided'}), 400

    try:
        with lock:
            result_vector = perform_operation(positive, negative)

            # Rebuild Faiss index if new words are added
            if any(word not in words for word in embeddings_dict.keys()):
                global faiss_index, words
                faiss_index, words = build_faiss_index(embeddings_dict)

            similar_words = find_most_similar_faiss(result_vector, faiss_index, words)
            formatted_results = [
                {'word': word, 'distance': float(distance)}
                for word, distance in similar_words
            ]
            return jsonify(formatted_results)
    except Exception as e:
        app.logger.error(f"Error in /api/operate: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
