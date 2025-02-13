# Raspberry Pi 5 AI Setup (Hybrid RAG) – Project Journal 🚀

This repository showcases a **Retrieval-Augmented Generation (RAG)** pipeline on a **Raspberry Pi 5** (8GB, anticipating future 16GB models). Over time, we've **iterated** on the design, focusing on **document chunking**, **embedding** with a smaller model, and **final text generation** using **Mistral 7B (Q4_K_M)**. This `README.md` evolves like a **journal**, tracking our updates, versions, and future goals.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Changelog & Versions](#changelog--versions)
3.  [Raspberry Pi 5 Photo & Diagram](#raspberry-pi-5-photo--diagram)
4.  [Hybrid RAG Pipeline](#hybrid-rag-pipeline)
5.  [Setup & Requirements](#setup--requirements)
6.  [Chunking & Tuning](#chunking--tuning)
7.  [Testing & Benchmarking](#testing--benchmarking)
8.  [Future / "Pro Version" Suggestions](#future--pro-version-suggestions)
9.  [Repo Structure & .gitignore](#repo-structure--gitignore)
10. [Credits & Contact](#credits--contact)
11. [Example Usage](#example-usage)

---

## 1. Project Overview
<a name="project-overview"></a>

Originally, we attempted to embed with **Mistral 7B** via `llama-cpp-python`. However, older builds only returned **1 float per token**, not a 4096-dimensional vector. Hence, we pivoted to:

-   **Sentence Transformers** for **embeddings** (e.g., `all-MiniLM-L6-v2`).
-   **Mistral 7B (Q4_K_M)** for **final text generation**.
-   **FAISS** for vector indexing.

**Google Drive** integration is used to **download** PDFs/TXT, which we parse, chunk, and store in the FAISS index. On queries, we embed the question with Sentence Transformers, retrieve top-k chunks, and feed them + the question to Mistral for the final answer. Authentication uses a service account (see `api-credentials.json` and `service_account_key.json` in `.gitignore`). We currently support PDF and TXT files, with basic error handling for download failures.

---

## 2. Changelog & Versions
<a name="changelog--versions"></a>

We maintain a **journal** of changes:

### **v1.0 (Initial Hybrid RAG) - 2024-02-29**

-   **Switched** to a smaller embedding model (`all-MiniLM-L6-v2`).
-   **Used** Mistral 7B strictly for generation.
-   **Implemented** `build_faiss_index.py` & `query_rag.py`.

### **v1.1 (Chunking & Tuning) - 2024-03-03**

-   Explored **chunk sizes** (300–800 chars).
-   Set overlap ~50 to avoid cutting context.
-   Documented best practices for balancing retrieval accuracy vs. indexing overhead.

### **v1.2 (Testing & Benchmarking) - 2024-03-07**

-   Added **timing code** in scripts to measure index build time.
-   Began tracking **query latency** (embed + FAISS retrieval + Mistral generation).
-   Noted CPU usage & memory constraints on Pi 5.

Future versions (see [Section 8](#future--pro-version-suggestions)) will incorporate more advanced features (UI front-end, advanced chunking logic, potential partial Hailo offload, etc.).

---

## 3. Raspberry Pi 5 Photo & Diagram
<a name="raspberry-pi-5-photo--diagram"></a>

### 3.1 Photo

Here's a **close-up photo** of the Raspberry Pi 5 in action (example):

![Raspberry Pi 5 Photo](images/rpi5_photo.jpg "Raspberry Pi 5")

*(Place the actual image `rpi5_photo.jpg` in an `images/` folder in your repo.)*

### 3.2 Minimal Pipeline Diagram

Below is a simple ASCII pipeline with **no extra fluff**:

\`\`\`

┌───────────┐ ┌────────────────┐ ┌─────────────────┐
│ Drive docs │ │ build\_faiss\_ │ │ query\_rag.py │
│ dl       │─→ │ index.py     │─→ │ (User query) │
└───────────┘ │ (Parse, chunk, │ │ embed + retrieve│
             │ embed, store  │ │ top chunks ->   │
             │ in FAISS)      │ │ Mistral 7B gen  │
└────────────────┘ └─────────────────┘

\`\`\`

---

## 4. Hybrid RAG Pipeline
<a name="hybrid-rag-pipeline"></a>

### 4.1 Basic Flow

1.  **Download**: `download_docs.py` fetches docs from Google Drive into `downloaded_files/`.
2.  **Build Index**: `build_faiss_index.py` parses docs, chunks them, embeds each chunk with Sentence Transformers, stores vectors in FAISS (saved in `faiss_index/`).
3.  **Query**: `query_rag.py` embeds the question, retrieves top chunks from FAISS, merges them into a prompt, and calls Mistral 7B for the final answer.

### 4.2 Why Sentence Transformers + Mistral?

-   **Sentence Transformers** → **fast CPU-based embeddings** (e.g., 384-dim).
-   **Mistral 7B** → advanced local generation (~3–4 tokens/s on Pi 5).

---

## 5. Setup & Requirements
<a name="setup--requirements"></a>

### 5.1 Raspberry Pi 5 Environment

-   **Pi 5** (8GB now, future 16GB).
-   **OS**: Debian Bookworm (64-bit).
-   **Python**: 3.11+

\`\`\`bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate
\`\`\`

> **Thermal Note:** Under sustained load, the Pi 5 CPU can hit 80–90°C. A fan or PoE hat is strongly recommended. In a pinch—especially in Finland—opening your balcony door for some fresh, cold air can quickly cool the device if you see 85°C+.

### 5.2 Install Dependencies

Create a `requirements.txt` file with:

\`\`\`
sentence-transformers
faiss-cpu
PyPDF2
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
llama-cpp-python
\`\`\`

Then install:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**llama-cpp-python Installation Note:** You might need specific build flags for your Pi 5. An example (disabling CUDA):

\`\`\`bash
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python
\`\`\`

### 5.3 Models

  - Sentence Transformers: `all-MiniLM-L6-v2` (downloaded automatically).
  - Mistral 7B (Q4\_K\_M): \~4.1 GB in .gguf, from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF), placed in `~/models/`.

-----

## 6\. Chunking & Tuning

<a name="chunking--tuning"></a>

### 6.1 Chunk Size & Overlap

  - Typical range: 300–800 chars, overlap \~50.
  - Smaller → more precise retrieval, bigger overhead.
  - Larger → fewer chunks, can degrade retrieval precision.

### 6.2 Advanced Splitting

  - Paragraph / heading splits for semantic coherence.
  - Use libraries like LangChain text splitters for token-based or paragraph-based logic.

### 6.3 top\_k & Similarity Threshold

  - Usually top-3–5. If incomplete, top-8–10.
  - Optionally filter out chunks below a similarity threshold.

-----

## 7\. Testing & Benchmarking

<a name="testing--benchmarking"></a>

### 7.1 Functional Tests

  - End-to-end: a small doc in `downloaded_files/`, build index, query with known question.
  - Confirm correct chunk retrieval & final Mistral answer.

### 7.2 Performance

  - Index Build time (chunking + embedding).
  - Query Latency (user question → final answer).
  - Memory & CPU usage at \~400% CPU, up to \~85°C.

### 7.3 Example Benchmarks

Below are example performance tables for LLM inference and FAISS:

**LLM Inference:**

| Mode        | Inference Time | RAM Used  | CPU Temp |
| ----------- | ------------- | -------- | -------- |
| CPU         | 5.79 s        | \~0.92 GB | \~75.7°C  |
| Hailo       | 5.85 s        | \~0.92 GB | \~76.3°C  |

(Times are approximate for a certain test. Actual values may differ on Pi 5 conditions.)

**Vector Search (FAISS):**

| Dataset       | Indexing Time | Query Speed      |
| ------------- | ------------- | ---------------- |
| Small Corpus  | 3.2 s         | 0.5 ms/query     |
| Large Corpus  | TBD           | TBD              |

(Again, example numbers showing the difference in scale. You can expand or replace with actual logs.)

-----

## 8\. Future / "Pro Version" Suggestions

<a name="future--pro-version-suggestions"></a>

Now that we have a stable pipeline, here are 10 advanced ideas:

1.  **Advanced Document Preprocessing:** semantic merges, OCR for scanned PDFs.
2.  **Alternative Vector Stores:** Milvus, Weaviate, or Chroma for advanced filtering.
3.  **UI/UX Enhancements:** Build a small FastAPI or Flask front-end.
4.  **Monitoring & Logging:** log each query, measure latency, memory usage, temperature.
5.  **Batch Embedding:** embed multiple chunks at once for faster index builds.
6.  **Hybrid Vector + Keyword:** combine simple keyword filtering with vector retrieval.
7.  **Partial Fine-Tuning:** consider LoRA for domain tasks on Mistral.
8.  **Hailo-8L Offload:** compile partial layers to `.hef` if feasible.
9.  **Automatic Summaries:** reduce chunk size by summarizing large docs.
10. **Multi-turn Chat:** store conversation context in memory for follow-up queries.

-----

## 9\. Repo Structure & .gitignore

<a name="repo-structure--gitignore"></a>

Recommended structure:

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

-----

## 10\. Credits & Contact

<a name="credits--contact"></a>

© 2025 – Built & tested by Richard Wirén.

**Contributing:**

Open PRs or Issues to:

  - Add new chunking logic or alternative embeddings
  - Integrate Hailo-8L offload
  - Add a minimal web GUI or multi-turn chat

-----

## 11\. Example Usage

<a name="example-usage"></a>

1.  Download documents:

    \`\`\`bash
    python download_docs.py
    \`\`\`

2.  Build the FAISS index:

    \`\`\`bash
    python build_faiss_index.py
    \`\`\`

3.  Query the RAG system:

    \`\`\`bash
    python query_rag.py "Your question here"
    \`\`\`

-----

Happy hacking—with a Pi 5 RAG pipeline that's carefully chunked, tested, and tested again under local embeddings (Sentence Transformers) + Mistral 7B for final generation. And if your CPU creeps up near 85°C, never forget: a swift blast of fresh Finnish air via your balcony door can do wonders to cool it down in a pinch\! 🚀
