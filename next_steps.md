# RPi5 Hybrid RAG Project - Next Steps

This document outlines the prioritized next steps for developing the Raspberry Pi 5 Hybrid RAG project. It's organized into three phases: Core Refinements, Incremental Improvements, and Advanced Features.

## I. Core Refinements (Highest Priority - Foundation)

These steps are crucial for creating a robust, maintainable, and easily configurable system.  Do these *before* adding new features.

### 1. Unified Code Structure, Error Handling, and Logging

**Goal:** Refactor the existing code for better organization, add comprehensive error handling, and implement detailed logging.

**Action Items:**

*   **Refactor into Functions:**
    *   Create a `utils.py` file (or similar) to house reusable functions.
    *   Define the following functions (and potentially others as needed):
        *   `fetch_docs_from_drive(folder_id=None, file_ids=None)`:  Handles Google Drive interaction.  Returns a list of downloaded file paths.
        *   `parse_document(filepath)`:  Opens and parses a single PDF or TXT file. Returns the raw text content. Includes error handling for file format issues.
        *   `chunk_text(text, chunk_size, overlap)`:  Implements character-based chunking. Returns a list of chunks.
        *   `embed_chunks(chunks, model_name)`:  Embeds a list of chunks using the specified Sentence Transformer model.  Returns a NumPy array of embeddings.  Handles batching internally (see Step 4).
        *   `build_faiss_index(embeddings, index_path)`:  Creates (or loads, see Step 5) a FAISS index, adds embeddings, and saves it.
        *   `query_faiss_index(query, index_path, top_k)`: Embeds the query, searches the FAISS index, returns top-k chunk indices and distances.
        *   `generate_answer(query, context, model_path)`: Formats the prompt and calls Mistral 7B.
    *   Modify `download_docs.py`, `build_faiss_index.py`, and `query_rag.py` to use these functions.

*   **Error Handling:**
    *   Wrap *all* file I/O, network operations (Google Drive), FAISS operations, and model loading/inference calls in `try-except` blocks.
    *   Catch *specific* exceptions (e.g., `FileNotFoundError`, `requests.exceptions.RequestException`, `faiss.Error`).  Avoid overly broad `except Exception:` blocks.
    *   Provide informative error messages to the user.

*   **Logging:**
    *   Use the Python `logging` module.
    *   Configure logging to write to a file (e.g., `rag_pipeline.log`) and optionally to the console.
    *   Use different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) appropriately.
    *   Include timestamps, filenames, chunk numbers (where relevant), error messages, and other useful information in log messages.
    * Consider adding a `get_logger()` function to utils.py.

**Deliverables:**

*   Refactored `download_docs.py`, `build_faiss_index.py`, `query_rag.py`.
*   `utils.py` with well-defined functions.
*   Comprehensive error handling.
*   Detailed logging.

### 2. Configuration File and Enhanced CLI

**Goal:** Centralize all configuration parameters and provide flexible command-line control.

**Action Items:**

*   **Configuration File:**
    *   Create a `config.yaml` file (recommended; use `PyYAML`).  Alternatively, use `config.ini` (with `configparser`) or `config.json` (with `json`).
    *   Define parameters for:
        *   Google Drive folder ID (or other authentication settings).
        *   Sentence Transformer model name (default: `all-MiniLM-L6-v2`).
        *   Mistral 7B model path (default: `~/models/mistral-7b-v0.1.Q4_K_M.gguf`).
        *   FAISS index path (default: `faiss_index/`).
        *   Chunk size (default: e.g., 500).
        *   Overlap (default: e.g., 50).
        *   `top_k` (default: e.g., 5).
        *   Logging level (default: INFO).
        *   Log file path (default: `rag_pipeline.log`).
    *   Load the configuration file in each script. Provide default values in the code that are overridden by the config file.

*   **Command-Line Arguments (`argparse`):**
    *   Use the `argparse` module in *each* script (`download_docs.py`, `build_faiss_index.py`, `query_rag.py`).
    *   Allow *all* configuration parameters to be overridden via command-line arguments. This provides a hierarchy: CLI arguments > config file > hardcoded defaults.
    *   `download_docs.py`:
        *   `--folder-id`: Google Drive folder ID.
        *   `--file-ids`: (Optional) Comma-separated list of specific file IDs.
        *   `--output-dir`: Where to save downloaded files (default: `downloaded_files/`).
        *   `--overwrite`: Flag to control overwriting existing files.
    *   `build_faiss_index.py`:
        *   `--input-dir`: Directory containing downloaded files.
        *   `--output-index`: Path to save the FAISS index.
        *   `--chunk-size`: Override the chunk size.
        *   `--overlap`: Override the overlap.
        *   `--embedding-model`: Override the embedding model.
        *   `--append`: Flag to add to an existing index (see Step 5).
    *   `query_rag.py`:
        *   `--index-path`: Path to the FAISS index.
        *   `--query`:  Directly provide the query text.
        *   `--top-k`: Override the number of chunks to retrieve.
        *   `--model-path`: Override the Mistral 7B model path.
        *   `--interactive`: Flag to enable/disable interactive mode (default: interactive).
        * `--show-context`: Flag to display retrieved chunks.

**Deliverables:**

*   `config.yaml` (or `.ini`/`.json`) file.
*   `argparse` implementation in all scripts.

### 3. Basic Document Preprocessing

**Goal:** Improve the quality of text before chunking and embedding.

**Action Items:**

*   Modify the `parse_document` function (in `utils.py`) to include the following:
    *   Remove excessive whitespace (multiple spaces, tabs, newlines).  Use regular expressions (`re.sub`).
    *   Handle special characters:
        *   Convert smart quotes/apostrophes to standard ASCII equivalents.
        *   Consider removing or replacing other non-alphanumeric characters (depending on your needs).
    *   (Optional) Remove HTML tags using `BeautifulSoup4` (if your documents might contain HTML).  Add `beautifulsoup4` to `requirements.txt` if you use this.
    *   (Optional) Convert text to lowercase.  This is *usually* beneficial for embedding models, but you should test it.  Make this a configurable option.

**Deliverables:**

*   Updated `parse_document` function with preprocessing logic.

## II. Incremental Improvements (Medium Priority)

These steps build upon the refined codebase and add significant functionality.

### 4. Batch Embedding

**Goal:** Speed up the FAISS index building process.

**Action Items:**

*   Modify `build_faiss_index.py` (or the `embed_chunks` function) to process chunks in batches.
*   Accumulate a list of chunks (up to a certain `batch_size`).
*   Call the Sentence Transformer's `encode()` method with the *list* of chunks: `embeddings = model.encode(list_of_chunks, batch_size=...)`.
*   Add the resulting batch of embeddings to the FAISS index.
*   Experiment with different `batch_size` values (e.g., 32, 64, 128) to find the optimal balance between speed and memory usage on the Pi 5.

**Deliverables:**

*   Modified `build_faiss_index.py` (or `embed_chunks` function) with batch embedding.
*   Notes on optimal `batch_size` for your setup.

### 5. FAISS Index Management (Appending)

**Goal:** Allow adding new documents to an existing FAISS index without rebuilding from scratch.

**Action Items:**

*   Modify `build_faiss_index.py`:
    *   If the `--append` flag is provided, load the existing FAISS index using `faiss.read_index(index_path)`.
    *   If `--append` is *not* provided, create a new index as before.
    *   Whether creating or appending, add the new embeddings using `index.add(embeddings)`.
    *   Save the (potentially updated) index using `faiss.write_index(index, index_path)`.

**Deliverables:**

*   Modified `build_faiss_index.py` with append functionality.

### 6. Alternative Embedding Models

**Goal:** Enable easy switching between different Sentence Transformer models.

**Action Items:**

*   Ensure the embedding model name is configurable via `config.yaml` and the `--embedding-model` CLI argument.
*   Test with at least one alternative model (e.g., `multi-qa-MiniLM-L6-cos-v1`, `all-mpnet-base-v2`).  Compare speed and retrieval quality.

**Deliverables:**

*   Confirmation that the system works with multiple embedding models.
*   Notes on the performance differences between models.

### 7. Enhanced Chunking (Paragraph Splits)

**Goal:** Improve chunking quality by splitting on paragraph boundaries.

**Action Items:**

*   Implement a `split_by_paragraph(text)` function in `utils.py`.
    *   Split the text based on blank lines (`\n\n`) or heading markers (e.g., `#`, `##` in Markdown).  Regular expressions are useful here.
*   Add a `--chunking-method` argument to `build_faiss_index.py` (and the config file) to choose between `character` (existing method) and `paragraph`.
*   Modify `build_faiss_index.py` to use the appropriate chunking function based on the selected method.

**Deliverables:**

*   `split_by_paragraph` function.
*   `--chunking-method` argument.
*   Comparison of retrieval results using character vs. paragraph chunking.

## III. User Interaction & Advanced Features (Lower Priority)

### 8. Interactive Query Mode (Refined)

**Goal:** Improve the user experience of the interactive query mode.

**Action Items:**

*   Modify `query_rag.py`:
    *   Use a `while True:` loop to continuously prompt for questions.
    *   Add a command to exit the loop (e.g., typing "exit" or "quit").
    *   (Optional) Add a `--show-context` flag. If enabled, print the retrieved chunks *before* displaying the final answer. This helps the user understand the reasoning.
    * (Optional) Consider using the `rich` library (add to `requirements.txt`) for colored output and better formatting, or `prompt_toolkit` for more advanced input handling.

**Deliverables:**

*   Updated `query_rag.py` with improved interactive mode.

### 9. Minimal Web UI (FastAPI)

**Goal:** Create a basic web interface for querying the system.

**Action Items:**

*   Create a new file, `app.py` (or similar).
*   Install FastAPI: `pip install fastapi uvicorn`.  Add `fastapi` and `uvicorn` to `requirements.txt`.
*   Implement a *single* endpoint:
    *   `POST /query` (or `GET /query` with a `text` parameter).
    *   This endpoint should:
        *   Receive the query text from the request.
        *   Call the necessary functions (`query_faiss_index`, `generate_answer`) to process the query.
        *   Return the final answer as JSON: `{"answer": "..."}`.
*   Create a *very* simple `index.html` file with a text input field and a submit button.  No need for complex JavaScript; a basic HTML form is sufficient.
*   Run the app using Uvicorn: `uvicorn app:app --reload --port 8000`.

**Deliverables:**

*   `app.py` with the FastAPI endpoint.
*   `index.html` with a basic query form.
*   Instructions on how to run the web UI.

### 10. Multi-turn Chat (Basic)

**Goal:** Enable basic follow-up questions.

**Action Items:**

*   Modify `query_rag.py`:
    *   Store the previous question and answer.  A simple list or string is sufficient for now.
    *   Before generating the answer to a new question, prepend the previous Q&A to the current query (or use a more sophisticated prompt template that incorporates the context).  Be mindful of Mistral 7B's context length limit.
* Add config option to choose number of turns.

**Deliverables:**
*    Modified `query_rag.py`.
*   A config or CLI param for controlling how many turns to keep

### 11. Hailo-8L Offloading (Exploratory)

**Goal:** Investigate the *possibility* of using a Hailo-8L accelerator.

**Action Items:**

*   **Research:** Start by thoroughly researching Hailo-8L compatibility with Sentence Transformers and Mistral 7B.  Look for existing examples, documentation, and tutorials.  This is a research-heavy step.
*   **Small Model Test:** If promising, try a *very small* model (e.g., a tiny BERT model) first to test the basic Hailo workflow (ONNX conversion, Hailo runtime).
*   **Incremental Approach:** If the small model works, gradually increase complexity.  Do *not* attempt to jump directly to Mistral 7B.

**Deliverables:**

*   Research notes on Hailo-8L compatibility.
*   (Potentially) a working example with a very small model.

### 12. Advanced Query Strategies

**Goal:** Improve the quality of answers by refining how the retrieved context is used.

**Action Items:**
*    Refine prompt templates.
*   Consider weighted chunk merging.

### 13. Fine-Tuning or Adapters
*   Consider LoRA.

### 14. Advanced Document Preprocessing (OCR)

**Goal:** Enable processing of scanned documents.

**Action Items:**

*   If you need to handle scanned documents, integrate an OCR library (e.g., Tesseract) into your `parse_document` function.

### 15. Alternative Vector Databases

**Goal:** Explore options beyond FAISS for scalability and features.

**Action Items:**
* Consider Milvus, Weaviate or Chroma

This `NEXT_STEPS.md` document provides a comprehensive and prioritized roadmap for your project. Remember to commit your code and update both your `README.md` and this `NEXT_STEPS.md` as you make progress. Good luck!
