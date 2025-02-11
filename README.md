# Raspberry Pi 5 AI Setup

This repository documents the setup process for running Generative AI, RAG, and FAISS on a **Raspberry Pi 5 (16GB)**.

Waiting for the AI hat...

## Overview

This project focuses on deploying **local LLMs, vector search, and AI-enhanced workflows** on the Raspberry Pi 5.
It includes configurations for:
- **Chatbot (General LLM):** Gemma-2B (Q4_K_M)
- **Summarization (RAG):** Mistral-7B (Q4_K_M)
- **Vector Search:** FAISS or ChromaDB

## Setup Steps

### 1️⃣ Initial Raspberry Pi 5 Configuration
- Updated system (`sudo apt update && sudo apt upgrade -y`)
- Installed necessary tools (`git`, `tmux`, `screen`, `htop`, `curl`)
- Created and configured user `riku` with **SSH key authentication**
- Expanded filesystem & configured locales

### 2️⃣ Python & Virtual Environment
- Installed `python3-venv`, `pip`, and set up **`ai_env` virtual environment**
  ```bash
  python3 -m venv ~/ai_env
  source ~/ai_env/bin/activate
  ```
- Installed AI-related dependencies:
  ```bash
  pip install faiss-cpu langchain transformers torch google-api-python-client google-auth
  ```

### 3️⃣ Download & Run LLMs
- Downloaded **Mistral-7B Q4_K_M** model (GGUF format)
- Configured Llama.cpp for **optimized inference on Raspberry Pi 5**
- Tested context size (`n_ctx`) scaling until **crash point identified at 30720**

## Logs & Testing
- **System logs** saved in [`rpi5_ai_setup_log.txt`](rpi5_ai_setup_log.txt)
- **Model test results** saved in `mistral_ctx_test.log`

## Next Steps 🚀
- Optimize performance for **longer context windows**
- Fine-tune **memory management** on Raspberry Pi 5
- Implement **vector search** with FAISS or ChromaDB
- Explore **Google Workspace API** integration

---

📌 *Contributions & improvements welcome!*# rpi5-ai-setup
Raspberry Pi 5 setup for Generative AI, FAISS, RAG, and local LLMs.
