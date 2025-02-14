# Raspberry Pi 5 AI (Hybrid RAG) – Project Journal 🚀

This `README.md` documents a **Retrieval-Augmented Generation (RAG)** pipeline built on a **Raspberry Pi 5 16GB**. 

The project uses:

-   **Sentence Transformers** (specifically `all-MiniLM-L6-v2`) for efficient text embeddings.
-   **Mistral 7B (Q4_K_M)** for local text generation.
-   **FAISS** for fast vector similarity search.
-   **Google Drive** as a document source (PDFs and TXT files).
-   And, when necessary, the "**fresh Finnish air**" method for rapid CPU cooling! 
-   Also installed a **BM2L-AIS-H8L** Hailo AI hat, however only minimal testing so far...

This file serves as a project journal, tracking progress, design decisions, and future plans.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Changelog & Versions](#changelog--versions)
3.  [Raspberry Pi 5 Photo & Pipeline Diagram](#rpi5-diagram)
4.  [Hybrid RAG Workflow](#rag-workflow)
5.  [Setup & Requirements](#setup)
6.  [Chunking & Tuning](#chunking)
7.  [Testing & Benchmarking](#testing)
8.  [Future suggestions](#pro-suggestions)
9.  [Credits & Contact](#credits)
10. [Example Usage](#usage)
11. [Next Steps](#next-steps)

---

## 1. Project Overview
<a name="project-overview"></a>

The initial goal was to use Mistral 7B for both embedding and generation. However, older versions of `llama-cpp-python` only provided a single float per token as output, not the full 4096-dimensional embedding vector.  Therefore, we adopted a *hybrid* approach:

-   **Sentence Transformers:**  Provides CPU-efficient text embeddings (384 dimensions).
-   **Mistral 7B (Q4_K_M):**  Handles the final text generation (achieving 3-4 tokens/second on the Pi 5).
-   **FAISS:** Enables fast vector indexing and similarity search.
-   **Google Drive:**  Serves as the source for documents (PDFs and TXT files), accessed via a service account.

The pipeline chunks documents, creates embeddings, stores them in FAISS, and then, at query time, embeds the user's question, retrieves the most relevant document chunks from FAISS, and feeds those chunks along with the question to Mistral 7B to generate a final answer.

---

## 2. Changelog & Versions

### **Development History**

*   **v1.0 (Initial Hybrid RAG) - 2024-02-29**
    *   Switched to `all-MiniLM-L6-v2` for embeddings.
    *   Used Mistral 7B for generation only.
    *   Implemented `build_faiss_index.py` and `query_rag.py`.

*   **v1.1 (Chunking & Tuning) - 2024-03-03**
    *   Explored chunk sizes and overlap.
    *   Documented best practices for chunking.

*   **v1.2 (Testing & Benchmarking) - 2024-03-07**
    *   Added timing code.
    *   Tracked query latency and resource usage.

*   **v2.0 (OCR, Refactoring, and Robustness) - 2024-02-13**
    *   **Added OCR support using `pytesseract` and `pdf2image`.** This allows the pipeline to handle scanned PDFs.
    *   **Refactored code into `utils.py`:** Improved modularity and maintainability.
    *   **Implemented a `config.yaml` file:** Centralized configuration parameters.
    *   **Added comprehensive error handling and logging:** Made the system more robust and easier to debug.
    *   **Implemented batch embedding:** Significantly sped up index creation.
    *   **Added a persistent chunk store (`chunk_store.json`):** Allows retrieval of original text chunks.
    *   **Improved command-line argument parsing:** Made scripts more flexible and user-friendly.
    *   **Added support for basic keyword filtering.**
    *   **Added basic multi-turn chat support.**

**Current Working Version: [v2.0 (located in the `version2` directory)](version2/)**

---

<a name="rpi-diagram"></a>
### 3.1 Pi 5 with AI Hat (beneath)

![Close-up of a Raspberry Pi 5](images/rpi5_photo.jpg "Raspberry Pi 5").

### 3.2 Pipeline Diagram

![Pipeline Diagram](images/vit-diagram.jpg "RAG Pipeline")

---

<a name="rag-workflow"></a>
## 4. Hybrid RAG Workflow (Version 2)

The workflow for the current version (v2.0, in the `version2` directory) is as follows:

1.  **Download (`download_docs.py`):**
    *   Authenticates with Google Drive using a service account (credentials stored separately and *not* committed to the repository).
    *   Fetches PDF and TXT files from a specified Google Drive folder (or specific file IDs, if provided via command-line arguments).
    *   Saves the downloaded files to the `downloaded_files/` directory.

2.  **Index (`build_faiss_index.py`):**
    *   Loads documents from the `downloaded_files/` directory.
    *   **Parses Documents:**
        *   For PDFs, it uses OCR (`pytesseract` and `pdf2image`) to extract text. This correctly handles *scanned* PDFs.
        *   For TXT files, it reads the text content directly.
        *   Performs basic text preprocessing (removes extra whitespace, etc.).
    *   **Chunks Text:** Splits the extracted text into smaller pieces.  Supports:
        *   **Character-based chunking:** Splits into chunks of a specified size (e.g., 500 characters) with a specified overlap (e.g., 50 characters).
        *   **Paragraph-based chunking:** Splits text based on paragraph boundaries.
    *   **Creates Embeddings:** Uses the specified Sentence Transformer model (default: `all-MiniLM-L6-v2`) to create embeddings for each chunk.  This is done in *batches* for efficiency.
    *   **Builds/Updates FAISS Index:** Creates a new FAISS index or appends to an existing one, storing the embeddings.  Saves the index to `faiss_index/index.faiss`.
    *   **Saves Chunk Data:** Saves the *original text* of each chunk, along with the *filepath* of the source document, to a JSON file (`chunk_store.json`). This is *critical* for retrieving the context for answer generation.

3.  **Query (`query_rag.py`):**
    *   Loads the FAISS index (`faiss_index/index.faiss`) and the chunk store (`chunk_store.json`).
    *   Embeds the user's query using the same Sentence Transformer model.
    *   **Searches FAISS:** Finds the top-k most similar chunk embeddings.
    *   **(Optional) Keyword Filtering:** If keywords are provided (via `--keywords`), filters the retrieved chunks to include only those containing *all* specified keywords (AND logic).
    *   **Retrieves Chunk Text:** Uses the indices from FAISS to retrieve the *original text* of the relevant chunks from `chunk_store.json`.
    *   **Generates Answer:** Constructs a prompt for Mistral 7B, including the retrieved context (chunks) and the user's query. Calls Mistral 7B to generate the final answer.
    *   **Interactive/Direct Mode:** Supports both interactive querying (via a prompt) and direct queries via command-line arguments (`--query`).
    *   **(Basic) Multi-turn Chat:** Can optionally retain a limited history of previous questions and answers to provide context for follow-up questions (`--context-turns`).

---

## 5. Setup & Requirements
<a name="setup"></a>
### 5.1 Pi 5 Environment

-   Raspberry Pi 5 (16GB RAM recommended)
-   Debian Bookworm (64-bit)
-   Python 3.11+

\`\`\`bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv ~/ai_env
source ~/ai_env/bin/activate
\`\`\`

> **Thermal Note:** The Raspberry Pi 5 CPU can reach temperatures of 80-90°C under sustained load.  Using a heatsink and fan, or a PoE hat with a fan, is highly recommended to prevent thermal throttling.

### 5.2 Dependencies & Models

Create a `requirements.txt` file with the following contents:

\`\`\`
sentence-transformers
faiss-cpu
PyPDF2
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
llama-cpp-python
\`\`\`

Install the dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**`llama-cpp-python` Installation:** You may need to specify build flags when installing `llama-cpp-python` on the Raspberry Pi 5.  For example, to disable CUDA (which is not available on the Pi), use:

\`\`\`bash
CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python
\`\`\`

Consult the `llama-cpp-python` documentation for other build options.

**Models:**

-   **Sentence Transformers:**  The project uses `all-MiniLM-L6-v2`. This model will be downloaded automatically by the `sentence-transformers` library.
-   Mistral 7B (Q4_K_M): ~4.1 GB in .gguf. Download directly from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf) and place it in `~/models/`..

---

## 6. Chunking & Tuning

-   **Chunk Size:** The recommended chunk size is between 300 and 800 characters, with an overlap of approximately 50 characters.  This balances retrieval precision and indexing overhead.
-   **`top_k`:**  The `top_k` parameter (number of chunks to retrieve) is typically set to 3-5.  If answers seem incomplete, increase `top_k` to 8-10.
-   **Advanced Chunking:**  For improved retrieval quality, consider splitting documents by paragraphs or headings, or using more sophisticated token-based splitting methods (e.g., with LangChain's text splitters).

---

## 7. Testing & Benchmarking
<a name="testing"></a>
### 7.1 Functional Tests

To perform a basic functional test:

1.  Place a small test document (PDF or TXT) in the `downloaded_files/` directory.
2.  Run `build_faiss_index.py` to create the FAISS index.
3.  Run `query_rag.py` with a question related to the content of the test document.
4.  Verify that the retrieved chunks are relevant and that Mistral 7B generates a correct answer.

### 7.2 Performance

Key performance metrics to track:

-   **Index Build Time:** The total time taken to chunk and embed the documents and build the FAISS index.
-   **Query Latency:** The time elapsed between submitting a query and receiving the final answer. This includes embedding the query, searching FAISS, and generating the answer with Mistral 7B.
-   **Memory and CPU Usage:** Monitor resource usage using tools like `top` or `htop`.

### 7.3 Example Benchmarks (Approximate)

**LLM Inference (Mistral 7B):**

| Mode   | Inference Time | RAM Used  | CPU Temp |
|--------|---------------|----------|----------|
| CPU    | 45.79 s       | ~0.92 GB | ~75.7°C  |
| Hailo  | 45.85 s       | ~0.92 GB | ~76.3°C  |

- The model used here was not optimal for the AI hat...

**FAISS Vector Search:**

| Dataset       | Indexing Time | Query Speed    |
|---------------|---------------|----------------|
| Small Corpus  | 3.2 s         | 0.5 ms/query   |
| Large Corpus  | TBD           | TBD            |

---

## 8. Future Suggestions
<a name="future"></a>

-  **Document Preprocessing:** Implement OCR for scanned PDFs and semantic paragraph merging.
-  **Alternative Vector Databases:** Explore Milvus, Weaviate, or Chroma for advanced features.
-  **Web UI:** Develop a simple web interface using FastAPI or Flask.
-  **Monitoring:** Implement logging and monitoring of resource usage (CPU, memory, temperature).
-  **Batch Embedding:** Process multiple chunks simultaneously during index building.
-  **Hybrid Search:** Combine vector search with keyword-based filtering.
-  **Fine-Tuning:** Explore fine-tuning Mistral 7B or using LoRA for domain-specific adaptation.
-  **Hailo-8L Offload:** Investigate offloading computation to a Hailo-8L accelerator (experimental).
-  **Automatic Summarization:** Summarize large chunks before feeding them to Mistral 7B.
-  **Multi-turn Chat:** Implement basic conversational context.
 
- see also [next steps](#next-steps)

---

## 9. Credits & Contact
<a name="project-overview"></a>
© 2025 – Built & tested by Richard (at) Wiren dot fi. (with some AI help...)

**Contributing**

Open PRs or Issues to:

-   Add new chunking or embedding methods.
-   Integrate Hailo-8L offloading.
-   Develop a web UI.

---

## 10. Example Usage
<a name="usage"></a>
1.  **Download Documents:**

    \`\`\`bash
    python download_docs.py --folder-id YOUR_GOOGLE_DRIVE_FOLDER_ID
    \`\`\`

    *(Replace `YOUR_GOOGLE_DRIVE_FOLDER_ID` with the actual ID. Omit `--folder-id` to use a default folder, if configured.)*

2.  **Build FAISS Index:**

    \`\`\`bash
    python build_faiss_index.py
    \`\`\`

    *(This processes downloaded documents and creates the FAISS index.)*

3.  **Query the System:**

    **Interactive Mode:**

    \`\`\`bash
    python query_rag.py  # Start the interactive query prompt
    \`\`\`
    *(Type your question at the prompt.)*

    **Direct Query:**

    \`\`\`bash
    python query_rag.py "What is the advantage of Mistral on Pi 5?"  # Run with a direct query
    \`\`\`

---

## 11. Next Steps
<a name="next-steps"></a>
See the [next_steps.md](next_steps.md) file for a draft roadmap of future development considerations.
