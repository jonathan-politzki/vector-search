# precompute_embeddings.py
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

def get_embedding(text, model='text-embedding-ada-002'):
    response = client.embeddings.create(input=text,
    model=model)
    embedding = response.data[0].embedding
    return embedding

def precompute_embeddings(vocabulary, output_file='embeddings.json'):
    embeddings = {}
    for word in vocabulary:
        embedding = get_embedding(word)
        embeddings[word] = embedding  # Embeddings are lists, JSON serializable
    with open(output_file, 'w') as f:
        json.dump(embeddings, f)

if __name__ == '__main__':
    vocabulary = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'doctor', 'nurse', 'brother', 'sister', 'husband', 'wife']  # Add more words as needed
    precompute_embeddings(vocabulary)
