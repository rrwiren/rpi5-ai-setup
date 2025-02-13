# Raspberry Pi 5 AI Setup (Hybrid RAG) – Project Journal 🚀

This single-file **journal** tracks our **Retrieval-Augmented Generation (RAG)** pipeline on a **Raspberry Pi 5** (8GB, with a future 16GB model). We use:

-   **Sentence Transformers** (e.g., `all-MiniLM-L6-v2`) for embeddings
-   **Mistral 7B (Q4_K_M)** for local text generation
-   **FAISS** for vector indexing
-   **Google Drive** for doc downloads (PDF/TXT)
-   Occasional "**fresh Finnish air**" method for emergency CPU cooling

Over time, we've iterated on chunking, memory usage, and local inference improvements. This `README.md` is our **journal**, capturing versions, best practices, and future expansions.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Changelog & Versions](#changelog--versions)
3.  [Raspberry Pi 5 Photo & Pipeline Diagram](#rpi5-diagram)
4.  [Hybrid RAG Workflow](#rag-workflow)
5.  [Setup & Requirements](#setup)
6.  [Chunking & Tuning](#chunking)
7.  [Testing & Benchmarking](#testing)
8.  [Retrospective & Next Steps](#retrospective)
9.  [Future / "Pro" Suggestions](#pro-suggestions)
10. [Repo Structure & .gitignore](#repo-structure)
11. [Credits & Contact](#credits)
12. [Example Usage](#usage)

---

<a name="project-overview"></a>

## 1. Project Overview

We initially tried letting **Mistral 7B** embed with older `llama-cpp-python`, but it only output **1 float per token** (not the 4096-d hidden state). Hence, the **hybrid** approach:

-   **Sentence Transformers** → CPU-friendly embeddings (~384 dims)
-   **Mistral 7B (Q4_K_M)** → final generation (3–4 tokens/s on Pi 5)
-   **FAISS** → vector indexing & retrieval
-   **Google Drive** → doc ingestion (via a service account JSON)

We chunk each doc, embed, store vectors in FAISS, then at query time, we embed the user question, retrieve top chunks, and feed them + question to Mistral for the final answer.

---

<a name="changelog--versions"></a>

## 2. Changelog & Versions

### **v1.0 (Initial Hybrid RAG) - 2024-02-29**

-   Switched to `all-MiniLM-L6-v2` for embeddings
-   Mistral 7B for generation only
-   Implemented `build_faiss_index.py`, `query_rag.py`

### **v1.1 (Chunking & Tuning) - 2024-03-03**

-   Explored chunk sizes (300–800 chars), overlap ~50
-   Balanced retrieval accuracy vs. indexing overhead

### **v1.2 (Testing & Benchmarking) - 2024-03-07**

-   Added timing code for index build & query latency
-   Noted Pi 5 CPU usage (~400%) & memory constraints
-   Observed Pi 5 hitting 85°C if used heavily

We plan to add more advanced features in future versions (UI front-end, partial Hailo offload, etc.).

---

<a name="rpi5-diagram"></a>

## 3. Raspberry Pi 5 Photo & Pipeline Diagram

### 3.1 Pi 5 Photo

![Raspberry Pi 5 Photo](images/rpi5_photo.jpg "Raspberry Pi 5")

### 3.2 Pipeline Diagram (Refined)

\`\`\`mermaid
graph LR
    A[Google Drive Docs] --> B(download_docs.py);
    B --> C(build_faiss_index.py);
    C --> D(FAISS Index);
    E[User Query] --> F(query_rag.py);
    F --> D;
    D --> F;
    F --> G(Mistral 7B);
    G --> H[Final Answer];
\`\`\`

This diagram uses [Mermaid.js](https://mermaid.js.org/).  GitHub natively supports Mermaid.

---

<a name="rag-workflow"></a>

## 4. Hybrid RAG Workflow

1.  **Download**: `download_docs.py` uses a service account credential to pull PDFs/TXT from Drive into `downloaded_files/`.
2.  **Index**: `build_faiss_index.py` parses & chunks docs, embeds each chunk (Sentence Transformers), and stores vectors in FAISS (`faiss_index/`).
3.  **Query**: `query_rag.py` takes a user question, embeds it, does top-k retrieval from FAISS, merges that context with the question, and calls Mistral 7B for a final answer.

Why this approach?

-   Mistral 7B is big (~4.1 GB Q4_K_M), but runs decently on Pi 5 for generation.
-   Sentence Transformers is a smaller CPU-based embedding model.

---

<a name="setup"></a>

## 5. Setup & Requirements

### 5.1 Pi 5 Environment

-   Pi 5 (8GB now, future 16GB)
-   Debian Bookworm (64-bit)
-   Python 3.11+

\`\`\`bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate
\`\`\`

> **Thermal Note:** Under sustained load, the Pi 5 CPU can hit 80–90°C. A fan or PoE hat is strongly recommended. In a pinch—especially in Finland—opening your balcony door for some fresh, cold air can quickly cool the device if you see 85°C+.

### 5.2 Dependencies & Models

Create `requirements.txt`:

\`\`\`
sentence-transformers
faiss-cpu
PyPDF2
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
llama-cpp-python
\`\`\`

Then:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**llama-cpp-python Installation Note:** On the Raspberry Pi 5, you might need to specify build flags to disable features that aren't supported or to optimize performance. For example, to disable CUDA (which isn't available on the Pi), you might use:

\`\`\`bash
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python
\`\`\`

Consult the `llama-cpp-python` documentation for other available flags.

### 5.3 Models

-   Sentence Transformers: `all-MiniLM-L6-v2` (downloaded automatically).
-   Mistral 7B (Q4_K_M): ~4.1 GB in .gguf. Download directly from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf) and place it in `~/models/`.

-----

## 6. Chunking & Tuning

<a name="chunking"></a>

-   **Chunk Size**: 300–800 chars recommended, overlap ~50
-   **Advanced**: break on paragraphs/headings or use token-based splits (LangChain, etc.)
-   **top_k**: 3–5 typically. If answers seem incomplete, go 8–10
-   If doc corpora are huge, chunk size might be bigger to limit overhead, but might degrade retrieval accuracy

---

<a name="testing"></a>

## 7. Testing & Benchmarking

<a name="testing"></a>

### 7.1 Functional Tests

-   Put a small doc in `downloaded_files/`, run `build_faiss_index.py`, then `query_rag.py`.
-   Confirm it retrieves correct chunks & Mistral outputs a relevant answer.

### 7.2 Performance

-   **Index Build** time: chunking + embedding overhead
-   **Query Latency**: embedding user question, FAISS search, Mistral generation
-   **Memory/CPU** usage: Pi 5 runs ~400% CPU, can exceed 80°C if heavily stressed

### 7.3 Example Benchmarks (Disclaimers: approximate data)

**LLM Inference:**

| Mode   | Inference Time | RAM Used  | CPU Temp |
|--------|---------------|----------|----------|
| CPU    |  45.79 s       | ~0.92 GB | ~75.7°C  |
| Hailo  |  45.85 s       | ~0.92 GB | ~76.3°C  |

**FAISS Vector Search:**

| Dataset       | Indexing Time | Query Speed    |
|---------------|---------------|----------------|
| Small Corpus  | 3.2 s         | 0.5 ms/query   |
| Large Corpus  | TBD           | TBD            |
---

<a name="retrospective"></a>

## 8. Retrospective & Next Steps

**What Went Well**:

-   Mistral 7B runs locally at 3–4 tokens/s on Pi 5, feasible for short queries
-   Sentence Transformers handles chunk embeddings quickly
-   No major memory issues if `chunk_size` is moderate

**Next Steps**:

-   Possibly test bigger `chunk_size` if indexing time becomes an issue
-   Evaluate concurrency or multi-threading for doc embedding
-   Try alternative models (like `multi-qa-MiniLM`) for different semantic needs

---

<a name="pro-suggestions"></a>

## 9. Future / "Pro" Suggestions

1.  **Advanced Preprocessing** (OCR, merges)
2.  **Alternative Vector DB** (Milvus, Weaviate, Chroma)
3.  **UI** (FastAPI/Flask front-end, multi-turn chat)
4.  **Monitoring** (Grafana dashboards for CPU usage, temps)
5.  **Batch Embedding** (faster index builds)
6.  **Hybrid Vector+Keyword** (combine boolean + vector)
7.  **Partial Fine-Tuning** (LoRA on Mistral or smaller model)
8.  **Hailo-8L Offload** (if partial layers can compile)
9.  **Auto Summaries** (reduce final chunk size for Mistral)
10. **Multi-turn Chat** (short memory of conversation context)

---

<a name="repo-structure"></a>

## 10. Repo Structure & .gitignore

\`\`\`
rpi5-ai-setup/
├── README.md
├── download_docs.py
├── build_faiss_index.py
├── query_rag.py
├── requirements.txt
├── images/
│   └── rpi5_photo.jpg
├── downloaded_files/
├── faiss_index/
└── ...
\`\`\`

`.gitignore`:

\`\`\`
api-credentials.json
service_account_key.json
downloaded_files/
faiss_index/
*.faiss
chunk_metadata.pkl
*.log
__pycache__/
\`\`\`
---

<a name="credits"></a>

## 11. Credits & Contact

© 2025 – Built & tested by Richard Wirén.

**Contributing**

  Open PRs or Issues to:

-   Add new chunking logic or alternative embeddings
-   Integrate Hailo-8L offload
-   Build a minimal web GUI or multi-turn chat

---

<a name="usage"></a>

## 12. Example Usage

1.  **Download** from Drive:

    \`\`\`bash
    python download_docs.py --folder-id YOUR_FOLDER_ID
    # or just 'python download_docs.py' if default is set
    \`\`\`

2.  **Build** the FAISS index:

    \`\`\`bash
    python build_faiss_index.py
    # watch for chunking logs, check final embeddings shape
    \`\`\`

3.  **Query** the pipeline:

    \`\`\`bash
    python query_rag.py
    # then type your question at prompt, or pass it directly:
    python query_rag.py "What is the advantage of Mistral on Pi 5?"
    \`\`\`

Happy hacking—embedding with **Sentence Transformers**, generating with **Mistral 7B**, occasionally **opening a Finnish balcony door** to keep the Pi 5 from overheating, and continuing to refine this pipeline as we learn more! 🚀
