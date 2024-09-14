# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import perform_operation, find_most_similar
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

@app.route('/api/operate', methods=['POST'])
@limiter.limit("10 per minute")
def operate():
    data = request.get_json()
    positive = data.get('positive', [])
    negative = data.get('negative', [])

    if not positive and not negative:
        return jsonify({'error': 'No words provided'}), 400

    try:
        logger.info(f"Received operation with positive: {positive}, negative: {negative}")
        result_vector = perform_operation(positive, negative)
        similar_words = find_most_similar(result_vector, top_n=5)
        logger.info(f"Operation successful. Returning results: {similar_words}")
        return jsonify(similar_words)
    except Exception as e:
        logger.error(f"Error in /api/operate: {e}", exc_info=True)
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