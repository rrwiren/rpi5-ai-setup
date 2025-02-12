import faiss
import numpy as np
import readline  # Enables command history & navigation
from llama_cpp import Llama

# Load the FAISS index
INDEX_PATH = "knowledge_base.index"
index = faiss.read_index(INDEX_PATH)

# Path to local Mistral-7B model
MODEL_PATH = "/home/riku/models/mistral-7b-v0.1.Q4_K_M.gguf"

# Load LLM with embeddings enabled
llm = Llama(model_path=MODEL_PATH, embedding=True, n_ctx=4096, verbose=False)

# Sample documents for context retrieval
documents = [
    "Raspberry Pi 5 AI setup guide",
    "How to install and configure Hailo-8L on Raspberry Pi",
    "Using FAISS for fast vector search",
    "Mistral-7B performance optimizations for edge AI",
    "Configuring power management for Raspberry Pi AI workloads"
]

# Function to generate embeddings
def get_embedding(text):
    return np.array(llm.embed(text), dtype=np.float32)

# Function to query knowledge base
def query_knowledge_base(query, top_k=3):
    query_vector = get_embedding(query)
    D, I = index.search(np.array([query_vector]), top_k)
    results = [documents[i] for i in I[0]]
    return results

# Function to generate AI responses
def generate_answer(query):
    retrieved_docs = query_knowledge_base(query)
    context = "\n".join(retrieved_docs)
    
    prompt = f"Based on the following documents:\n{context}\n\nAnswer the question: {query}"
    response = llm(prompt)
    
    return response["choices"][0]["text"]

# Interactive CLI loop
def chat():
    print("\n🤖 AI Assistant CLI Chatbot (RAG-Powered)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("📝 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting AI Assistant. See you next time!")
            break
        
        # Generate AI response
        answer = generate_answer(query)
        print(f"\n🤖 AI: {answer}\n")

if __name__ == "__main__":
    chat()

# Updates to be applied...
