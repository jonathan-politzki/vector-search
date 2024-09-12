from flask import Flask, request, jsonify
from flask_cors import CORS
from embedding_operations import load_model, perform_operation, find_similar_words

app = Flask(__name__)
CORS(app)

model = load_model()

@app.route('/api/operate', methods=['POST'])
def operate():
    data = request.json
    positive = data.get('positive', [])
    negative = data.get('negative', [])
    
    result_vector = perform_operation(model, positive, negative)
    similar_words = find_similar_words(model, result_vector)
    
    return jsonify(similar_words)

if __name__ == '__main__':
    app.run(debug=True)