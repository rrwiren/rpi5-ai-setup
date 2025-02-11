import numpy as np
import faiss
from llama_cpp import Llama

# Paths
MODEL_PATH = "/home/riku/models/mistral-7b-v0.1.Q4_K_M.gguf"
EMBEDDING_SIZE = 4096  # Fixed size for FAISS

# Load Llama model with embeddings enabled
print(f"🔄 Loading model from: {MODEL_PATH} ...")
llm = Llama(model_path=MODEL_PATH, n_ctx=16384, embedding=True, verbose=False)

# Sample documents
documents = [
    "Raspberry Pi 5 AI setup",
    "Mistral LLM integration",
    "FAISS for RAG",
    "Hailo-8 AI acceleration",
    "Running LLMs on edge devices"
]

# Function to get text embeddings with enforced size
def get_embedding(text):
    embedding = llm.embed(text)

    # Ensure embedding is a 1D NumPy array
    embedding = np.array(embedding, dtype=np.float32).flatten()

    # Normalize to fixed EMBEDDING_SIZE
    if embedding.shape[0] > EMBEDDING_SIZE:
        embedding = embedding[:EMBEDDING_SIZE]  # Truncate
    elif embedding.shape[0] < EMBEDDING_SIZE:
        padding = np.zeros(EMBEDDING_SIZE - embedding.shape[0], dtype=np.float32)
        embedding = np.concatenate((embedding, padding))  # Pad

    return embedding

# Generate document embeddings
vectors = np.array([get_embedding(doc) for doc in documents])
print(f"✅ Final embedding shape: {vectors.shape}")

# Create FAISS index
index = faiss.IndexFlatL2(EMBEDDING_SIZE)
index.add(vectors)
print("✅ FAISS Index Created & Documents Indexed!")

# Query FAISS
query_vector = get_embedding("AI project on Raspberry Pi")
D, I = index.search(query_vector.reshape(1, -1), 1)  # Ensure query vector is 2D

# Print best match
print(f"🔍 Best Match: {documents[I[0][0]]} (Distance: {D[0][0]})")
