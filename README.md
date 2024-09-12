# Embedding Space Navigator

Embedding Space Navigator is a web application that allows users to explore word embeddings and perform vector operations on them. Users can input positive and negative words to navigate the embedding space and find similar words based on these operations.

## Features

- Interactive web interface for inputting positive and negative words
- Backend API for performing embedding operations
- Utilizes OpenAI's text embedding model
- Efficient similarity search using FAISS
- Precomputed embeddings for common words to improve performance

## Tech Stack

### Frontend
- React
- Material-UI
- Axios for API calls

### Backend
- Flask
- Flask-CORS for handling CORS
- OpenAI API for generating embeddings
- FAISS for similarity search
- NumPy for vector operations

## Project Structure

```
embedding-space-navigator/
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── InputForm.js
│   │   │   └── ResultDisplay.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
│
├── backend/
│   ├── app.py
│   ├── embedding_operations.py
│   ├── precompute_embeddings.py
│   └── requirements.txt
│
├── embeddings.json
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/embedding-space-navigator.git
   cd embedding-space-navigator
   ```

2. Set up the backend:
   ```
   cd backend
   pip install -r requirements.txt
   ```
   Create a `.env` file in the backend directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Precompute embeddings:
   ```
   python precompute_embeddings.py
   ```

4. Set up the frontend:
   ```
   cd ../frontend
   npm install
   ```

5. Start the backend server:
   ```
   cd ../backend
   python app.py
   ```

6. Start the frontend development server:
   ```
   cd ../frontend
   npm start
   ```

7. Open your browser and navigate to `http://localhost:3000` to use the application.

## Usage

1. Enter positive words in the "Positive Words" input field, separated by commas.
2. Enter negative words in the "Negative Words" input field, separated by commas.
3. Click the "Calculate" button to perform the embedding operation.
4. View the results, which show the most similar words based on the vector operation.

## Future Improvements

- Add user authentication and personal word lists
- Implement more advanced embedding operations
- Visualize the embedding space using dimensionality reduction techniques
- Optimize performance for larger vocabulary sizes
- Add support for phrases and sentences

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.