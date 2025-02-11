import faiss
import numpy as np
from llama_cpp import Llama

# 🔄 Load Mistral-7B Model
MODEL_PATH = "/home/riku/models/mistral-7b-v0.1.Q4_K_M.gguf"
TARGET_DIM = 4096  # 🔥 Ensure all embeddings have exactly this size

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_batch=512, embedding=True, verbose=False)

# ✅ Function to Generate Fixed-Size Embeddings
def get_embedding(text):
    """Generates embeddings and ensures a fixed size of TARGET_DIM."""
    embedding = np.array(llm.embed(text), dtype=np.float32)

    if embedding.shape[0] < TARGET_DIM:
        # 🚀 Pad with zeros if too short
        embedding = np.pad(embedding, (0, TARGET_DIM - embedding.shape[0]), mode='constant')
    elif embedding.shape[0] > TARGET_DIM:
        # ✂ Truncate if too long
        embedding = embedding[:TARGET_DIM]

    return embedding.reshape(1, -1)  # 🔄 Ensure a strict 2D shape

# 📄 List of Documents
documents = [
    "Raspberry Pi 5 AI setup",
    "Mistral LLM integration",
    "FAISS for RAG",
    "Hailo-8 AI acceleration",
    "Running LLMs on edge devices"
]

# 🔄 Generate Embeddings
vectors = np.vstack([get_embedding(doc) for doc in documents])  # 🔥 vstack ensures 2D shape

# 🔍 Create FAISS Index
index = faiss.IndexFlatL2(TARGET_DIM)
index.add(vectors)

# 🧐 Query the Index with a New Input
query_text = "How to accelerate AI on Raspberry Pi?"
query_embedding = get_embedding(query_text)  # 🔥 Reshaped to (1, TARGET_DIM)

# 🔍 Search FAISS for the 3 Closest Matches
D, I = index.search(query_embedding, k=3)

# 📊 Print Results
print("\n✅ FAISS Vector Search Results:")
for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
    print(f"{rank+1}. Match: '{documents[idx]}' (Distance: {dist:.4f})")
