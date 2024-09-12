# Embedding Space Navigator

This project allows users to explore word embeddings through a web interface.

## Setup

1. Backend:
   - Navigate to the `backend` directory
   - Create a virtual environment: `python -m venv venv`
   - Activate the virtual environment:
     - On Unix or MacOS: `source venv/bin/activate`
     - On Windows: `venv\Scripts\activate`
   - Install dependencies: `pip install -r requirements.txt`
   - Download a pre-trained word embedding model and update the path in `embedding_operations.py`

2. Frontend:
   - Navigate to the `frontend` directory
   - Install dependencies: `npm install`

## Running the Application

1. Start the backend server:
   - In the `backend` directory, run: `flask run`

2. Start the frontend development server:
   - In the `frontend` directory, run: `npm start`

3. Open your browser and navigate to `http://127.0.0.1:5000`

## Next Steps

- Implement error handling and input validation
- Add styling to the frontend
- Optimize backend performance
- Add advanced features like visualizations
- Prepare for production deployment