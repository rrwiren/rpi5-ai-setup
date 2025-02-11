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
- **Model test results** saved in [`mistral_ctx_test.log`](mistral_ctx_test.log)

## Next Steps 🚀
- Optimize performance for **longer context windows**
- Fine-tune **memory managemen

# Raspberry Pi 5 AI Setup

## Overview
This repository documents the setup of Raspberry Pi 5 for Generative AI, FAISS, RAG, and local LLMs.
## Setup Steps
1. **System Preparation**
   - Installed Raspberry Pi OS (64-bit)
   - Configured SSH and user `user`
   - Expanded filesystem and updated system packages

2. **Essential Tools Installed**
   - `git`, `htop`, `tmux`, `screen`, `curl`
   - Python environment (`python3-venv`, `pip`)
   - FAISS, LangChain, and Transformers

3. **Local LLMs Setup**
   - Installed `llama.cpp`
   - Downloaded **Mistral-7B (Q4_K_M)** for RAG/Summarization
   - Configured **Gemma-2B (Q4_K_M)** for chatbot
   - Enabled **FAISS vector search**

## Logs & Testing
- **System logs**: 
- **Model test results**: `mistral_ctx_test.log`
- 

## Performance Optimization
- **Configured Zswap**
  - `zswap.enabled=1 zswap.compressor=lzo`
  - Verified with `stored_pages` and `pool_total_size`
- **Memory & Swap Management**
  - Swappiness set to `100` for testing, will optimize further
  - Swap space usage confirmed with `free -h`

## Next Steps 🚀
- Optimize performance for longer context windows
- Fine-tune memory management on Raspberry Pi 5
- Implement vector search with FAISS or ChromaDB
- Prepare for **AI Hat integration**

## Repository
[GitHub Repo](https://github.com/rrwiren/rpi5-ai-setup)

---

📌 **Contributions & Feedback Welcome!**

