# utils.py
import logging
import re
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from PyPDF2 import PdfReader
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO
import pytesseract  # Import pytesseract
from pdf2image import convert_from_path  # Import convert_from_path
from PIL import Image  # Import Image
import json
from llama_cpp import Llama # ADDED


def get_logger(name, log_level=logging.INFO, log_file="rag_pipeline.log"):
    """Configures and returns a logger (avoids duplicate handlers)."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:  # Only add handlers if they don't exist yet
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file, with error handling."""
    logger = get_logger(__name__)  # Use logger inside function
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config or {}  # Return empty dict if config is empty/None
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except Exception as e:
        logger.exception(f"Error loading config: {e}")  # More detailed logging
        return {}


def fetch_docs_from_drive(folder_id, service, file_ids=None):
    """Fetches documents from Google Drive and saves them locally."""
    logger = get_logger(__name__)
    downloaded_files = []

    try:
        if file_ids:  # Download specific files
            for file_id in file_ids:
                file_metadata = service.files().get(fileId=file_id).execute()
                file_name = file_metadata.get('name')
                request = service.files().get_media(fileId=file_id)
                fh = BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                with open(Path("downloaded_files") / file_name, "wb") as f:
                    f.write(fh.getvalue())
                downloaded_files.append(str(Path("downloaded_files") / file_name))  # Store the path
                logger.info(f'Downloaded: {file_name}')

        else:  # Download all files from the folder
            results = service.files().list(q=f"'{folder_id}' in parents", fields="nextPageToken, files(id, name)").execute()
            items = results.get('files', [])
            if not items:
                logger.warning('No files found in the specified Google Drive folder.')
                return []
            for item in items:
                file_id = item['id']
                file_name = item['name']
                request = service.files().get_media(fileId=file_id)
                fh = BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                with open(Path("downloaded_files") / file_name, "wb") as f:
                    f.write(fh.read())
                downloaded_files.append(str(Path("downloaded_files") / file_name)) # Store the path
                logger.info(f'Downloaded: {file_name}')
            logger.info(f"Downloaded {len(items)} files from Google Drive folder: {folder_id}")

    except Exception as e:
        logger.error(f"Error downloading files from Google Drive: {e}")
        return []  # Return empty list on error (important!)

    return downloaded_files  # Return the list of downloaded file *paths*

def parse_document(filepath):
    """Parses a PDF or TXT document, performing OCR if necessary."""
    logger = get_logger(__name__)
    text_content = ""
    try:
        filepath_obj = Path(filepath)
        if filepath_obj.suffix.lower() == ".pdf":
            with open(filepath_obj, "rb") as f:
                try:
                    # First, try PyPDF2 for text extraction
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    logger.debug(f"Initial PyPDF2 extraction (first 500 chars):\n{text_content[:500]}")

                except Exception as e:
                    logger.info(f"PyPDF2 failed on {filepath_obj}: {e}")

                # If PyPDF2 fails (or extracts very little text), try OCR
                if not text_content.strip():
                    logger.info(f"Attempting OCR on {filepath_obj}...")
                    try:
                        images = convert_from_path(filepath_obj)  # Convert PDF to images
                        for image in images:
                            text_content += pytesseract.image_to_string(image) + "\n"
                        logger.info(f"OCR successful on {filepath_obj}")
                    except Exception as e:
                        logger.error(f"OCR failed on {filepath_obj}: {e}")
                        return ""  # Return empty string on OCR failure

        elif filepath_obj.suffix.lower() == ".txt":
            with open(filepath_obj, "r", encoding="utf-8") as f:
                text_content = f.read()
        else:
            logger.warning(f"Unsupported file type: {filepath_obj.suffix}")
            return ""

        # --- Basic Preprocessing ---
        text_content = re.sub(r'\s+', ' ', text_content)  # Normalize whitespace
        text_content = text_content.strip()
        return text_content  # Return the extracted and cleaned text

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return ""
    except Exception as e:
        logger.error(f"Error parsing document {filepath}: {e}")
        return ""

def chunk_text(text, chunk_size, overlap):
    """Splits the text into chunks with specified size and overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def split_by_paragraph(text):
    """Splits the text into chunks by paragraph."""
    return text.split("\n\n")  # Split on blank lines

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Embeds a list of text chunks using a Sentence Transformer model."""
    logger = get_logger(__name__)
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        return np.array([])

def build_faiss_index(embeddings, index_path):
    """Builds (or loads and updates) a FAISS index."""
    logger = get_logger(__name__)
    dimension = embeddings.shape[1]
    try:
        if Path(index_path).exists():
            logger.info(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))
            index.add(embeddings)
        else:
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            index = faiss.IndexFlatL2(dimension)  # Or another suitable index type
            index.add(embeddings)
        faiss.write_index(index, str(index_path))
        logger.info(f"FAISS index built/updated and saved to {index_path}")
    except Exception as e:
        logger.exception(f"Error building/updating FAISS index: {e}")  # Use exception for stack trace
        raise  # Re-raise the exception to halt execution

def query_faiss_index(query_embedding, index_path, top_k):
    """Queries the FAISS index and returns distances and indices."""
    logger = get_logger(__name__)
    try:
        index = faiss.read_index(str(index_path))
        distances, indices = index.search(np.array([query_embedding]), top_k)
        return distances[0], indices[0]  # Return results for the single query
    except Exception as e:
        logger.exception(f"Error querying FAISS index: {e}")
        raise

def generate_answer(query, context, model_path):
    """Generates an answer using Mistral 7B."""
    logger = get_logger(__name__)
    try:
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4) #Increased n_ctx
        prompt = f"""Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query}
        Answer:
        """
        # Truncate the prompt if it exceeds the context window
        if len(prompt) > 4096:
            logger.warning(f"Prompt length ({len(prompt)}) exceeds context window (4096). Truncating.")
            prompt = prompt[:4096]

        output = llm(prompt, max_tokens=256, stop=["</s>"], echo=False) # Increased tokens
        answer = output['choices'][0]['text']
        return answer.strip()
    except Exception as e:
        logger.exception(f"Error generating answer with Mistral: {e}")
        raise

def load_chunk_store(chunk_store_path):
    """Loads the chunk store from a JSON file."""
    logger = get_logger(__name__)
    try:
        with open(chunk_store_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Chunk store file not found: {chunk_store_path}. Returning empty list.")
        return []  # Return empty list if file not found
    except Exception as e:
        logger.error(f"Error loading chunk store from {chunk_store_path}: {e}")
        return []

def get_chunk_text(index, chunk_store):
    """Retrieves chunk text from the loaded chunk store."""
    try:
        return chunk_store[index]['text']
    except IndexError:
        get_logger(__name__).error(f"Index {index} out of range in chunk store.")
        return ""
    except KeyError:
        get_logger(__name__).error(f"Chunk at index {index} does not have a 'text' key.")
        return ""
    except Exception as e:
        get_logger(__name__).error(f"Unexpected error retrieving chunk text: {e}")
        return ""
