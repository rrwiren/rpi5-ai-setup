# query_rag.py
import argparse
from utils import get_logger, load_config, embed_chunks, query_faiss_index, generate_answer
from llama_cpp import Llama
import logging
import json  # Use JSON for loading chunk data
from pathlib import Path

def main():
    """Queries the RAG system, either interactively or with a direct query."""

    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("--index-path", type=str, default="faiss_index/index.faiss", help="Path to the FAISS index.")
    parser.add_argument("--query", type=str, help="Direct query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--model-path", type=str, default="~/models/mistral-7b-v0.1.Q4_K_M.gguf", help="Path to the Mistral 7B model.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive query mode.")
    parser.add_argument("--show-context", action="store_true", help="Show the retrieved context before the answer.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument("--keywords", type=str, nargs='+', help="Keywords for hybrid search (optional).")  # New keywords argument
    parser.add_argument("--context-turns", type=int, default=0, help="Number of previous Q&A turns to include in context.")
    parser.add_argument("--chunk-store", type=str, default="chunk_store.json", help="Path to the chunk store file.") #Added
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)

    # Override config with CLI arguments
    index_path = args.index_path if args.index_path else config.get("index_path", "faiss_index/index.faiss")
    top_k = args.top_k if args.top_k else config.get("top_k", 5)
    model_path = args.model_path if args.model_path else config.get("model_path", "~/models/mistral-7b-v0.1.Q4_K_M.gguf")
    log_level_str = args.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    interactive = args.interactive if args.interactive is not None else config.get("interactive", True)  # Corrected default handling
    show_context = args.show_context
    keywords = args.keywords  # Get keywords from CLI
    context_turns = args.context_turns
    chunk_store_path = args.chunk_store if args.chunk_store else config.get("chunk_store", "chunk_store.json")

    # --- Logger Setup ---
    logger = get_logger(__name__, log_level=log_level)
    logger.info("Starting RAG query process...")
    logger.debug(f"Using configuration: {args}")

     # --- Load Embedding Model ---
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")

    # --- Load Mistral 7B Model ---
    try:
        # Expand the tilde (~) in the model path to the user's home directory
        expanded_model_path = str(Path(model_path).expanduser())
        llm = Llama(model_path=expanded_model_path, n_ctx=4096, n_threads=4)  # Moved to utils.py
    except Exception as e:
        logger.error(f"Error loading Mistral model: {e}")
        return

     # --- Load Chunk Store ---
    chunk_store = []
    try:
        with open(chunk_store_path, 'r', encoding='utf-8') as f:  # Specify UTF-8 and use json.load
            chunk_store = json.load(f)
        logger.info(f"Loaded {len(chunk_store)} chunks from {chunk_store_path}")
    except FileNotFoundError:
        logger.error(f"Chunk store file not found: {chunk_store_path}")
        return  # Exit if chunk store is missing
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in {chunk_store_path}.  Is it valid JSON?")
        return
    except Exception as e:
        logger.error(f"Error loading chunk store: {e}")
        return
    # --- Main Query Loop (Interactive or Direct) ---

    # Initialize conversation history
    conversation_history = []

    if interactive:
        print("Interactive query mode. Type your question or ':quit' to exit.")
        while True:
            query = input("Query: ")
            if query.lower() == ":quit":
                break

            # Add current query to conversation history
            conversation_history.append({"role": "user", "content": query})

            # Build context string from conversation history (last 'context_turns' turns)
            context_string = " ".join(
                [f"{turn['role']}: {turn['content']}" for turn in conversation_history[-context_turns:]]
            )
            query_with_context = context_string + " " + query


            try:
                query_embedding = embed_chunks([query_with_context], embedding_model_name)[0]
                distances, indices = query_faiss_index(query_embedding, index_path, top_k)
            except Exception as e:
                logger.error(f"Error during embedding/retrieval: {e}")
                print("An error occurred during embedding or retrieval.")
                continue  # Skip to the next iteration of the loop

            # --- Keyword Filtering (if keywords are provided) ---
            filtered_indices = []
            filtered_distances = []
            if keywords:
                keywords_lower = [k.lower() for k in keywords]
                for i, idx in enumerate(indices):
                    chunk_info = chunk_store[idx] # Access the dictionary
                    chunk_text = chunk_info['text'].lower() # Get text and lowercase

                    if all(keyword in chunk_text for keyword in keywords_lower):  # AND logic
                        filtered_indices.append(idx)
                        filtered_distances.append(distances[i])
                indices = filtered_indices
                distances = filtered_distances

            if show_context:
                logger.info("Retrieved Chunks (Indices):")
                for i in indices:
                     chunk_info = chunk_store[i]
                     logger.info(f"- Index: {i}, File: {chunk_info['filepath']}, Text (first 100 chars): {chunk_info['text'][:100]}...")


            if indices.size == 0: # Corrected empty array check
                print("No relevant chunks found.")
                continue

            context = " ".join([chunk_store[i]['text'] for i in indices]) # Get chunk text from store
            try:
                answer = generate_answer(query, context, expanded_model_path) #Use expanded path
                conversation_history.append({"role": "assistant", "content": answer}) #Add to history
                print(f"Answer: {answer}")

            except Exception as e:
                logger.error(f"Error during answer generation: {e}")
                print("An error occurred during answer generation.")
                continue

    elif args.query:  # Direct query from command line
        try:
            query_embedding = embed_chunks([args.query], embedding_model_name)[0]
            distances, indices = query_faiss_index(query_embedding, index_path, top_k)
        except Exception as e:
            logger.error(f"Error during embedding/retrieval: {e}")
            print("An error occurred during embedding or retrieval.")
            return
        # --- Keyword Filtering (if keywords are provided) ---
        filtered_indices = []
        filtered_distances = []
        if keywords:
            keywords_lower = [k.lower() for k in keywords]
            for i, idx in enumerate(indices):
                chunk_info = chunk_store[idx]  # Access the dictionary
                chunk_text = chunk_info['text'].lower() # Get text and make it lowercase
                if all(keyword in chunk_text for keyword in keywords_lower):  # AND logic
                    filtered_indices.append(idx)
                    filtered_distances.append(distances[i])
            indices = filtered_indices
            distances = filtered_distances

        if show_context:
            logger.info("Retrieved Chunks (Indices):")
            for i in indices:
                chunk_info = chunk_store[i]
                logger.info(f"- Index: {i}, File: {chunk_info['filepath']}, Text (first 100 chars): {chunk_info['text'][:100]}...")

        if indices.size == 0: # Corrected empty array check
            print("No relevant chunks found.")
            return

        context = " ".join([chunk_store[i]['text'] for i in indices])  # Get chunk text from store

        try:
            answer = generate_answer(args.query, context, expanded_model_path) # Use expanded path
            print(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            print("An error occurred during answer generation.")
            return
    else:
        logger.error("No query provided. Use --query or --interactive.")
        parser.print_help()  # Show help message if no query provided


if __name__ == "__main__":
    main()
