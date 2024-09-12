# embedding_operations.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import json
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Load precomputed embeddings
with open('embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)

# Convert lists to numpy arrays
embeddings_dict = {word: np.array(embedding) for word, embedding in embeddings_dict.items()}

def get_embedding(text, model='text-embedding-ada-002'):
    response = client.embeddings.create(input=text,
    model=model)
    embedding = response.data[0].embedding
    return np.array(embedding)

def perform_operation(positive_words, negative_words):
    # Fetch embeddings for all words
    positive_embeddings = []
    for word in positive_words:
        if word in embeddings_dict:
            positive_embeddings.append(embeddings_dict[word])
        else:
            embedding = get_embedding(word)
            positive_embeddings.append(embedding)
            embeddings_dict[word] = embedding  # Optionally cache the embedding

    negative_embeddings = []
    for word in negative_words:
        if word in embeddings_dict:
            negative_embeddings.append(embeddings_dict[word])
        else:
            embedding = get_embedding(word)
            negative_embeddings.append(embedding)
            embeddings_dict[word] = embedding  # Optionally cache the embedding

    # Ensure there is at least one embedding
    if positive_embeddings or negative_embeddings:
        dimension = len(next(iter(embeddings_dict.values())))
        positive_sum = np.sum(positive_embeddings, axis=0) if positive_embeddings else np.zeros(dimension)
        negative_sum = np.sum(negative_embeddings, axis=0) if negative_embeddings else np.zeros(dimension)

        # Compute result vector
        result_vector = positive_sum - negative_sum
        return result_vector
    else:
        raise ValueError("No embeddings found for the provided words.")

def build_faiss_index(embeddings_dict):
    words = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values())).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, words

def find_most_similar_faiss(result_vector, faiss_index, words, top_n=5):
    result_vector = np.array([result_vector]).astype('float32')
    distances, indices = faiss_index.search(result_vector, top_n)
    similar_words = [(words[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return similar_words
