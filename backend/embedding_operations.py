# backend/embedding_operations.py

import numpy as np
import faiss
import openai
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load precomputed embeddings
with open('embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)

# Convert lists to numpy arrays
embeddings_dict = {word: np.array(embedding) for word, embedding in embeddings_dict.items()}

def get_embedding(text, model='text-embedding-3-small'):
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response['data'][0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Error getting embedding for '{text}': {str(e)}")
        return None

def perform_operation(positive_words, negative_words):
    # Fetch embeddings for all words
    positive_embeddings = [embeddings_dict.get(word) or get_embedding(word) for word in positive_words]
    negative_embeddings = [embeddings_dict.get(word) or get_embedding(word) for word in negative_words]

    # Remove None values (failed embeddings)
    positive_embeddings = [emb for emb in positive_embeddings if emb is not None]
    negative_embeddings = [emb for emb in negative_embeddings if emb is not None]

    # Update embeddings_dict with new embeddings
    for word, emb in zip(positive_words + negative_words, positive_embeddings + negative_embeddings):
        if word not in embeddings_dict and emb is not None:
            embeddings_dict[word] = emb

    # Ensure there is at least one embedding
    if positive_embeddings or negative_embeddings:
        dimension = len(next(iter(embeddings_dict.values())))
        positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(dimension)
        negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(dimension)

        # Compute result vector
        result_vector = positive_sum - negative_sum
        return result_vector
    else:
        raise ValueError("No valid embeddings found for the provided words.")

def build_faiss_index(embeddings_dict):
    words = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values())).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, words

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    result_vector = np.array([result_vector]).astype('float32')
    distances, indices = faiss_index.search(result_vector, top_n)
    return [(words[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
