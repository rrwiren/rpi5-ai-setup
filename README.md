# Raspberry Pi 5 AI Setup (Hybrid RAG) – Project Journal 🚀

This single-file **journal** covers our **Retrieval-Augmented Generation (RAG)** pipeline on a **Raspberry Pi 5** (8GB, eventually 16GB). We use:

- **Sentence Transformers** (e.g., `all-MiniLM-L6-v2`) for embeddings
- **Mistral 7B (Q4_K_M)** for local text generation (~3–4 tokens/s)
- **FAISS** for vector indexing
- **Google Drive** for doc ingestion (PDF/TXT)
- Occasional **Finnish balcony** cooling if CPU temps spike

We've iterated on chunking, memory usage, local inference, and other optimizations. This `README.md` is our **journal**, tracking versions, best practices, and expansions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Changelog & Versions](#changelog--versions)
3. [Raspberry Pi 5 Photo & Pipeline Diagram](#rpi5-diagram)
4. [Hybrid RAG Workflow](#rag-workflow)
5. [Setup & Requirements](#setup)
6. [Chunking & Tuning](#chunking)
7. [Testing & Benchmarking](#testing)
8. [Retrospective & Next Steps](#retrospective)
9. [Future / "Pro" Suggestions](#pro-suggestions)
10. [Credits & Contact](#credits)
11. [Example Usage](#usage)

---

\```<a name="project-overview"></a>\```
## 1. Project Overview

We initially tried letting **Mistral 7B** embed with older `llama-cpp-python`, but it returned **1 float per token** (not the 4096-d hidden state). Hence, a **hybrid** approach:

- **Sentence Transformers** → CPU-friendly embeddings (~384-d).  
- **Mistral 7B** → advanced local generation on Pi 5.  
- **FAISS** → storing chunk embeddings for retrieval.  
- **Google Drive** → doc ingestion.

Each doc is chunked, embedded, stored in FAISS. For queries, we embed the user question, retrieve top chunks, feed them + question to Mistral for a final answer.

---

\```<a name="changelog--versions"></a>\```
## 2. Changelog & Versions

### **v1.0 (Initial Hybrid RAG) - 2024-02-29**
- Switched to `all-MiniLM-L6-v2` for embeddings  
- Mistral 7B for generation  
- Implemented `build_faiss_index.py`, `query_rag.py`

### **v1.1 (Chunking & Tuning) - 2024-03-03**
- Explored chunk sizes (300–800 chars), overlap ~50  
- Balanced retrieval accuracy vs. indexing overhead

### **v1.2 (Testing & Benchmarking) - 2024-03-07**
- Added timing code for index build + query latency  
- Observed Pi 5 CPU usage ~400%, up to 85°C  
- Memory constraints if chunking is too large

More features planned (UI front-end, partial Hailo offload, advanced chunking, etc.).

---

\```<a name="rpi5-diagram"></a>\```
## 3. Raspberry Pi 5 Photo & Pipeline Diagram

### 3.1 Pi 5 Photo
![Raspberry Pi 5 Photo](images/rpi5_photo.jpg "Raspberry Pi 5")

### 3.2 Pipeline Diagram (Using Provided Image)
Instead of an ASCII diagram, here’s a **screenshot** with the pipeline flow:

![RAG Pipeline Diagram](images/pipeline_diagram.png "RAG Pipeline")  
*(Replace with your actual image path if needed.)*

---

\```<a name="rag-workflow"></a>\```
## 4. Hybrid RAG Workflow

1. **Download**: `download_docs.py` fetches docs from Drive into `downloaded_files/`.  
2. **Index**: `build_faiss_index.py` parses & chunks docs, embeds them via Sentence Transformers, stores vectors in FAISS.  
3. **Query**: `query_rag.py` takes user Q, embeds, retrieves top-k chunks, merges them + question → Mistral final text.

---

\```<a name="setup"></a>\```
## 5. Setup & Requirements

### 5.1 Pi 5 Environment
- Pi 5 (8GB, 16GB in future)  
- Debian Bookworm (64-bit)  
- Python 3.11+  

\```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate
\```

**Memory & Thermal**:
- If near 8GB usage, consider zswap or a swapfile.  
- If CPU hits 85°C, use a **fan** or PoE hat, or open a **Finnish balcony door** in cold weather for a quick fix.

### 5.2 Dependencies & Models

Make `requirements.txt`:

\```
sentence-transformers
faiss-cpu
PyPDF2
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
llama-cpp-python
\``

Then:

\```bash
pip install -r requirements.txt
\```

**llama-cpp-python**: On Pi 5, might need:

\```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python
\```

**Models**:
- Sentence Transformers: `all-MiniLM-L6-v2`
- Mistral 7B (Q4_K_M): ~4.1 GB `.gguf` from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF), placed in `~/models/`.

---

\```<a name="chunking"></a>\```
## 6. Chunking & Tuning

- **Chunk size**: ~300–800 chars, overlap ~50  
- For large corpora, chunk_size can be bigger but might degrade retrieval precision  
- top_k: 3–5 typically; 8–10 if answers are incomplete  
- Advanced splits: paragraphs/headings or token-based (LangChain, etc.)

---

\```<a name="testing"></a>\```
## 7. Testing & Benchmarking

### 7.1 Functional Tests
- Put a doc in `downloaded_files/`, run `build_faiss_index.py`, then `query_rag.py`
- Check correctness of chunk retrieval & Mistral’s final answer

### 7.2 Performance
- **Index Build**: chunking + embedding overhead
- **Query Latency**: embedding user Q + FAISS retrieval + Mistral generation
- **Memory/CPU**: ~400% CPU usage, up to ~85°C if not cooled

### 7.3 Example Benchmarks

**LLM Inference**:

| Mode  | Inference Time | RAM Used  | CPU Temp |
|-------|---------------|-----------|----------|
| CPU   | 45.79 s        | ~0.92 GB  | ~75.7°C  |
| Hailo | 45.85 s        | ~0.92 GB  | ~76.3°C  |

**Vector Search (FAISS)**:

| Dataset       | Indexing Time | Query Speed    |
|---------------|---------------|--------------- |
| Small Corpus  | 3.2 s         | 0.5 ms/query   |
| Large Corpus  | TBD           | TBD            |

---

\```<a name="retrospective"></a>\```
## 8. Retrospective & Next Steps

**What Went Well**:
- Mistral 7B runs at 3–4 tokens/s on Pi 5, workable for short queries
- Sentence Transformers easily handles doc embeddings
- Memory usage is manageable with moderate chunk_size

**Next Steps**:
- Possibly bigger chunk_size if indexing time becomes large
- Evaluate concurrency for doc embedding
- Try alternative embedding models for specialized domains

---

\```<a name="pro-suggestions"></a>\```
## 9. Future / "Pro" Suggestions

1. **Doc Preprocessing**: OCR merges for scanned PDFs  
2. **Alternate Vector DB**: Milvus, Weaviate, Chroma  
3. **UI**: small FastAPI or Flask front-end  
4. **Monitoring**: CPU/memory dashboards  
5. **Batch Embedding**: embed multiple chunks at once  
6. **Hybrid Vector+Keyword**: combine boolean + vector  
7. **Partial Fine-Tuning**: LoRA on Mistral 7B  
8. **Hailo-8L Offload**: partial layers -> `.hef` if feasible  
9. **Auto Summaries**: reduce chunk text for Mistral  
10. **Multi-turn Chat**: short conversation context

---

\```<a name="credits"></a>\```
## 10. Credits & Contact

© 2025 – Built & tested by Richard Wirén.

**Contributing**  
Open PRs or Issues to:
- Add new chunking logic or alternative embeddings
- Integrate Hailo-8L offload
- Build a minimal web GUI or multi-turn chat

---

\```<a name="usage"></a>\```
## 11. Example Usage

**1. Download** from Drive:
\```bash
python download_docs.py --folder-id YOUR_FOLDER_ID
# or just 'python download_docs.py' if default folder is set
\```

**2. Build** the FAISS index:
\```bash
python build_faiss_index.py
# watch logs, confirm final embeddings shape
\```

**3. Query** the pipeline:
\```bash
python query_rag.py
# Then type your question, or pass it:
python query_rag.py "How does Mistral 7B compare on Pi 5?"
\```

That’s it—embedding with **Sentence Transformers**, generating with **Mistral 7B**, sometimes opening a **Finnish balcony door** if CPU temps spike, and evolving this pipeline as we progress! 🚀
