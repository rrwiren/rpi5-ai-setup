# Raspberry Pi 5 AI (Hybrid RAG) – Project Journal 🚀

This single-file **journal** tracks our **Retrieval-Augmented Generation (RAG)** pipeline on a **Raspberry Pi 5** (8GB, with a future 16GB model). We use:

-   **Sentence Transformers** (e.g., `all-MiniLM-L6-v2`) for embeddings
-   **Mistral 7B (Q4_K_M)** for local text generation
-   **FAISS** for vector indexing & retrieval
-   **Google Drive** to fetch docs (PDF/TXT)
-   Occasionally “**fresh Finnish air**” as a quick CPU-cooling hack

We've iterated on chunking, memory usage, local inference, and other optimizations. This `README.md` is our **journal**, showing versions, best practices, and future expansions.

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

## 1. Project Overview

We initially tried letting **Mistral 7B** embed with older `llama-cpp-python`, but it only output **one float per token**, missing the 4096-d hidden state. Hence, our **hybrid** approach:

-   **Sentence Transformers** → CPU-friendly embeddings (~384-dim).
-   **Mistral 7B (Q4_K_M)** → final generation (3–4 tokens/s on Pi 5).
-   **FAISS** → vector indexing/retrieval.
-   **Google Drive** → doc ingestion (via a service account).

Docs are chunked, embedded, stored in FAISS. At query time, we embed the user question, retrieve top chunks, pass them + question to Mistral for the final answer.

---

## 2. Changelog & Versions

### **v1.0 (Initial Hybrid RAG) - 2024-02-29**

-   Switched to `all-MiniLM-L6-v2` for embeddings
-   Mistral 7B for generation
-   Implemented `build_faiss_index.py` / `query_rag.py`

### **v1.1 (Chunking & Tuning) - 2024-03-03**

-   Explored chunk sizes (300–800 chars), overlap ~50
-   Balanced retrieval accuracy vs. indexing overhead

### **v1.2 (Testing & Benchmarking) - 2024-03-07**

-   Timing code for index build & query latency
-   Observed CPU usage ~400% on Pi 5, up to 85°C
-   Memory constraints if chunking is large

More features planned (UI front-end, partial Hailo offload, advanced chunking, etc.).

---

## 3. Raspberry Pi 5 Photo & Pipeline Diagram

### 3.1 Pi 5 Photo

![Raspberry Pi 5 Photo](images/rpi5_photo.jpg "Raspberry Pi 5")

### 3.2 Pipeline Diagram (Plain ASCII, Improved)

\`\`\`
+---------------------+     +------------------------+     +---------------------+
|  Google Drive       | --> |  build_faiss_index.py  | --> |    query_rag.py     |
|  (PDFs, TXT)        |     |  (Parse, Chunk, Embed,  |     |    (User Query)     |
+---------------------+     |   Store in FAISS)       |     +---------------------+
                            +------------------------+                   |
                                    ^                                      |
                                    |                                      V
                            +------------------------+     +---------------------+
                            |     FAISS Index        |     |      Mistral 7B     |
                            | (Chunk Embeddings)     | <-- |    (Generation)     |
                            +------------------------+     +---------------------+
                                                                            |
                                                                            V
                                                                    +-----------------+
                                                                    |  Final Answer   |
                                                                    +-----------------+
\`\`\`

Key changes to the diagram:

*   **More Consistent Boxes:**  Used consistent box sizes and shapes.
*   **Clearer Arrows:**  Used `-->` for consistent directional arrows.  The up arrow from FAISS to `query_rag.py` shows the retrieval process.
*   **More Descriptive Labels:** Added "(Chunk Embeddings)" to the FAISS Index box.
*    **Alignment**: Better vertical and horizontal alignment.

---

## 4. Hybrid RAG Workflow

1.  **Download**: `download_docs.py` pulls docs from Drive into `downloaded_files/`.
2.  **Index**: `build_faiss_index.py` parses & chunks docs, embeds them (Sentence Transformers), stores in FAISS.
3.  **Query**: `query_rag.py` takes user Q, embeds, retrieves top chunks, merges with question → Mistral 7B final answer.

---

## 5. Setup & Requirements

### 5.1 Pi 5 Environment

-   Pi 5 (8GB now, 16GB expected)
-   Debian Bookworm (64-bit)
-   Python 3.11+

\`\`\`bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate
\`\`\`

**Memory & Temp Tips**:

-   If near memory limit, consider **zswap** or a swapfile.
-   If CPU hits 85°C, use a **fan** or PoE hat, or in a pinch, open a **Finnish balcony door** for fresh air.

### 5.2 Dependencies & Models

Make `requirements.txt`:

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

**llama-cpp-python**: On Pi 5, you might do:

\`\`\`bash
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python
\`\`\`

**Models**:

-   Sentence Transformers: `all-MiniLM-L6-v2`
-   Mistral 7B (Q4_K_M): ~4.1 GB in .gguf, from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF). Place in `~/models/`.

---

## 6. Chunking & Tuning

-   **Chunk size**: ~300–800 chars, overlap ~50
-   If docs are huge, bigger `chunk_size` reduces overhead but might degrade retrieval precision
-   `top_k`: 3–5 usually; if incomplete, 8–10
-   Advanced: break on paragraphs/headings or token-based splits

---

## 7. Testing & Benchmarking

### 7.1 Functional Tests

-   Put a known doc in `downloaded_files/`, run `build_faiss_index.py`, then `query_rag.py`
-   Check correctness of chunk retrieval & final Mistral answer

### 7.2 Performance

-   **Index Build** time: chunking + embedding overhead
-   **Query Latency**: embedding question + FAISS retrieval + Mistral generation
-   **Memory/CPU** usage: ~400% CPU on Pi 5, can reach 85°C

### 7.3 Example Benchmarks

**LLM Inference**:

| Mode     | Inference Time | RAM Used | CPU Temp |
|----------|---------------|----------|----------|
| CPU      | 45.79 s       | ~0.92 GB | ~75.7°C  |
| Hailo    | 45.85 s       | ~0.92 GB | ~76.3°C  |

**FAISS Vector Search**:

| Dataset        | Indexing Time | Query Speed    |
|----------------|---------------|----------------|
| Small Corpus   | 3.2 s         | 0.5 ms/query   |
| Large Corpus   | TBD           | TBD            |

---

## 8. Retrospective & Next Steps

**What Went Well**:

-   Mistral 7B runs at 3–4 tokens/s on Pi 5, feasible for short queries
-   Sentence Transformers handles doc embeddings quickly
-   Memory usage is manageable with moderate `chunk_size`

**Next Steps**:

-   Possibly test bigger `chunk_size` if indexing time is an issue
-   Evaluate concurrency for doc embedding
-   Try alternative embedding models (e.g. `multi-qa-MiniLM-L6-cos-v1`)

---

## 9. Future / "Pro" Suggestions

1.  **Doc Preprocessing**: OCR, merges
2.  **Alternate Vector DB**: Milvus, Weaviate, Chroma
3.  **UI**: small FastAPI/Flask front-end
4.  **Monitoring**: dashboards for CPU/memory usage
5.  **Batch Embedding**: embed multiple chunks at once
6.  **Hybrid Vector+Keyword**: combine boolean + vector
7.  **Partial Fine-Tuning**: LoRA on Mistral 7B
8.  **Hailo-8L Offload**: partial layers -> `.hef` if feasible
9.  **Auto Summaries**: reduce final chunk size
10. **Multi-turn Chat**: short conversation context

---

## 10. Repo Structure & .gitignore

\`\`\`text
rpi5-ai-setup/
├── README.md               # This file
├── download_docs.py        # Fetches documents
├── build_faiss_index.py    # Builds the FAISS index
├── query_rag.py           # Handles queries
├── requirements.txt      # Python dependencies
├── images/
│   └── rpi5_photo.jpg     # Photo of the Pi 5
├── downloaded_files/       # (Gitignored) Downloaded docs
└── faiss_index/            # (Gitignored) FAISS index files
\`\`\`

Key improvements to the Repo Structure:

*   **Plain Text:**  Instead of trying to use Markdown code block formatting (which can be inconsistent for file trees), I've used plain text.  This is *much* more reliable for readability.
*   **Comments:** Added short comments explaining the purpose of each file/directory.  This makes the structure self-documenting.
*   **Consistent Indentation:** Used consistent indentation to clearly show the hierarchy.
*  **Simplified**:

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

## 11. Credits & Contact

© 2025 – Built & tested by Richard Wirén.

**Contributing**

Open PRs or Issues to:

-   Add new chunking logic or alternative embeddings
-   Integrate Hailo-8L offload
-   Build a minimal web GUI or multi-turn chat

---

## 12. Example Usage

1.  **Download** from Drive:

    \`\`\`bash
    python download_docs.py --folder-id YOUR_FOLDER_ID
    # or just 'python download_docs.py' if default is set
    \`\`\`

2.  **Build** the FAISS index:

    \`\`\`bash
    python build_faiss_index.py
    # watch for logs, confirm final embeddings shape
    \`\`\`

3.  **Query** the pipeline:

    \`\`\`bash
    python query_rag.py
    # Type your question, or pass it directly:
    python query_rag.py "What is Mistral's advantage on Pi 5?"
    \`\`\`

Happy hacking—embedding with **Sentence Transformers**, generating with **Mistral 7B**, occasionally opening a **Finnish balcony door** to keep CPU temps in check, and evolving this pipeline as we learn more! 🚀
