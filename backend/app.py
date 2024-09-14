from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import perform_operation, find_most_similar
import threading
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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
            result_vector = perform_operation(positive, negative)

            if result_vector is None:
                return jsonify({'error': 'Failed to compute result vector'}), 500

            similar_words = find_most_similar(result_vector, top_n=5)
            
            if not similar_words:
                return jsonify({'error': 'Failed to find similar words'}), 500

            formatted_results = [
                {'word': word, 'distance': float(distance)}  # Ensure distance is JSON serializable
                for word, distance in similar_words
            ]
            logger.info(f"Operation successful. Returning results: {formatted_results}")
            return jsonify(formatted_results)
    except ValueError as e:
        logger.error(f"Error in /api/operate: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error("Unexpected error in /api/operate", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

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
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))