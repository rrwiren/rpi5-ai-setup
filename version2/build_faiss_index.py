# build_faiss_index.py
import argparse
from pathlib import Path
import numpy as np
from utils import (get_logger, load_config, parse_document, chunk_text,
                   embed_chunks, build_faiss_index, split_by_paragraph)
import logging
import json  # Import the json module


def main():
    """Builds or updates a FAISS index from documents in a directory."""

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Build or update a FAISS index from text documents.")
    parser.add_argument("--input-dir", type=str, default="downloaded_files", help="Directory containing input documents.")
    parser.add_argument("--output-index", type=str, default="faiss_index/index.faiss", help="Path to save the FAISS index.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters.")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name.")
    parser.add_argument("--append", action="store_true", help="Append to an existing index instead of rebuilding.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument("--chunking-method", type=str, default="character", choices=["character", "paragraph"], help="Chunking method: 'character' or 'paragraph'.")
    parser.add_argument("--chunk-store", type=str, default="chunk_store.json", help="Path to save the chunk and filepath information.")

    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)

    # Override config with CLI arguments (if provided)
    chunk_size = args.chunk_size if args.chunk_size else config.get("chunk_size", 500)
    overlap = args.overlap if args.overlap else config.get("overlap", 50)
    embedding_model_name = args.embedding_model if args.embedding_model else config.get("embedding_model", "all-MiniLM-L6-v2")
    input_dir = args.input_dir if args.input_dir else config.get("input_dir", "downloaded_files")
    output_index_path = args.output_index if args.output_index else config.get("output_index", "faiss_index/index.faiss")
    log_level_str = args.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    chunking_method = args.chunking_method if args.chunking_method else config.get("chunking_method", "character")
    chunk_store_path = args.chunk_store if args.chunk_store else config.get("chunk_store", "chunk_store.json") # Get chunk_store path


    # --- Logger Setup ---
    logger = get_logger(__name__, log_level=log_level)
    logger.info("Starting FAISS index building process...")
    logger.debug(f"Using configuration: {args}")

    # --- Input Directory Check ---
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_dir_path}")
        return

    # --- Create output directory ---
    output_index_path_obj = Path(output_index_path)
    output_index_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # --- Initialize chunk store ---
    all_chunks = []
    filepaths = [] # List for filepaths

    # --- Main Processing Loop ---
    for filepath in input_dir_path.glob("*"):  # Iterate through all files
        if filepath.is_file():
            logger.info(f"Processing file: {filepath}")
            text_content = parse_document(filepath)
            if text_content:
                if chunking_method == "character":
                    chunks = chunk_text(text_content, chunk_size, overlap)
                elif chunking_method == "paragraph":
                    chunks = split_by_paragraph(text_content)
                logger.info(f"  Extracted {len(chunks)} chunks.")
                all_chunks.extend(chunks)
                # Store filepath for each chunk associated with this file.
                filepaths.extend([str(filepath)] * len(chunks))



    if not all_chunks:
        logger.warning("No chunks extracted from documents. Exiting.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_chunks(all_chunks, embedding_model_name)  # Pass chunks directly

    if embeddings.size == 0:
        logger.error("Embedding failed. Exiting.")
        return

    logger.info(f"Building/Updating FAISS index...")
    build_faiss_index(embeddings, output_index_path)

    logger.info("FAISS index building/updating complete.")

    # --- Save chunk store and filepaths---
    chunk_data = []
    for chunk, filepath in zip(all_chunks, filepaths):
        chunk_data.append({'filepath': filepath, 'text': chunk})  # Create a dictionary

    with open(chunk_store_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4)  # Use json.dump for proper JSON formatting

if __name__ == "__main__":
    main()
