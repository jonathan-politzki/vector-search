# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import perform_operation, find_most_similar
import threading
import logging
from dotenv import load_dotenv
import os
import numpy as np
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for API token
API_TOKEN = os.getenv('API_TOKEN')
if not API_TOKEN:
    raise ValueError("API_TOKEN is not set in the .env file")

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Lock for thread safety
lock = threading.Lock()

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
            result_vector, word_embeddings = perform_operation(positive, negative)
            
            if result_vector is None:
                return jsonify({'error': 'Failed to compute result vector. Please try again later.'}), 500

            logger.info(f"Result vector shape: {result_vector.shape}, norm: {np.linalg.norm(result_vector)}")
            logger.info(f"Number of word embeddings: {len(word_embeddings)}")

            similar_words = find_most_similar(result_vector, word_embeddings)
            
            logger.info(f"Similar words found: {similar_words}")

            formatted_results = [
                {
                    'word': word,
                    'similarity': float(similarity),
                    'is_input': word in positive or word in negative
                }
                for word, similarity in similar_words
            ]

            result = {
                'results': formatted_results,
                'message': 'Operation successful.'
            }
            logger.info("Operation successful. Returning results.")
            return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error in /api/operate: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500
            
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Perform a simple operation to check if the system is working
        result_vector, _ = perform_operation(['test'], [])
        if result_vector is not None and np.linalg.norm(result_vector) != 0:
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'Failed to perform test operation'}), 500
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
