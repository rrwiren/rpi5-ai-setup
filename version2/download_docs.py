# download_docs.py
import argparse
from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from utils import get_logger, load_config, fetch_docs_from_drive  # Import the function
import io
import logging

def main():
    """Downloads documents from a Google Drive folder."""

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download documents from Google Drive.")
    parser.add_argument("--folder-id", type=str, help="Google Drive folder ID.")
    parser.add_argument("--file-ids", type=str, nargs='+', help="List of specific file IDs to download.")
    parser.add_argument("--output-dir", type=str, default="downloaded_files", help="Output directory for downloaded files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)
    # Override config with CLI arguments
    folder_id = args.folder_id if args.folder_id else config.get("google_drive_folder_id")
    file_ids = args.file_ids if args.file_ids else config.get("file_ids", [])
    output_dir = args.output_dir if args.output_dir else config.get("output_dir", "downloaded_files")
    log_level_str = args.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)  # Default to INFO


    # --- Logger Setup ---
    logger = get_logger(__name__, log_level=log_level)
    logger.info("Starting document download process...")
    logger.debug(f"Using configuration: {args}")

    # --- Google Drive Authentication ---
    try:
        creds = service_account.Credentials.from_service_account_file(
            config["service_account_file"], scopes=config["scopes"]
        )
        service = build('drive', 'v3', credentials=creds)

    except Exception as e:
        logger.error(f"Error authenticating with Google Drive: {e}")
        return

     # --- Create Output Directory ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Download Files ---
    # Corrected function call:
    fetch_docs_from_drive(folder_id, service, file_ids)  # Call the imported function
    logger.info("Document download complete.")

if __name__ == "__main__":
    main()
