# Raspberry Pi 5 AI Setup 🚀

This project focuses on deploying **local LLMs, vector search, and AI-enhanced workflows** on the **Raspberry Pi 5**.
It includes optimizations for running **efficient** AI models, leveraging **Hailo-8L AI acceleration**, and integrating **vector search**.

---

## 🏗️ Project Overview

### ✅ Current AI Stack:
| **Component**              | **Model/Technology**       | **Purpose**                          |
|----------------------------|----------------------------|--------------------------------------|
| **LLM (Chatbot)**          | Gemma-2B (Q4_K_M)          | General LLM responses                |
| **RAG (Summarization)**    | Mistral-7B (Q4_K_M)        | Retrieval-augmented generation       |
| **Vector Search**          | FAISS / ChromaDB           | Efficient search & retrieval         |
| **Hardware Acceleration**  | Hailo-8L AI Hat            | Offloading AI tasks                  |

---

## 🔧 Setup & Configuration

### 1️⃣ Raspberry Pi 5 Setup:
- Installed **Raspberry Pi OS (64-bit)**
- Set up **SSH**, locales, and expanded filesystem.
- Configured **Zswap & memory optimizations** (`zswap.enabled=1`, `vm.swappiness=100`).
- Optimized **power & thermal management**.

### 2️⃣ Local LLM Inference:
- Installed `llama-cpp-python` to run **Mistral-7B (Q4_K_M)**.
- Successfully tested **n_ctx=16384** as a stable max context window.

### 3️⃣ Vector Search:
- **FAISS** `1.10.0` installed and configured.
- Successfully **indexed embeddings** & retrieved relevant search results.

### 4️⃣ Hailo-8L AI Accelerator:
- Installed `hailortcli` and confirmed **firmware 4.20.0** is working.
- Hailo-8L identified as **Hailo-8L AI ACC M.2 A+E KEY MODULE EXT TMP**.
- Some **firmware config files failed to load**, but basic functionality is available.
- Need to test **LLM offloading** on Hailo.

---

## 🛠️ Benchmarks

### **LLM Inference:**
| **Mode**  | **Inference Time** | **RAM Used** | **CPU Temp** |
|-----------|--------------------|-------------|-------------|
| **CPU**   | 45.79 sec          | 0.92 GB     | 75.7°C      |
| **Hailo** | 45.85 sec          | 0.92 GB     | 76.3°C      |

### **Vector Search (FAISS):**
| **Dataset**    | **Indexing Time** | **Query Speed** |
|----------------|-------------------|-----------------|
| Small Corpus   | 3.2 sec          | 0.5 ms/query    |
| Large Corpus   | TBD              | TBD             |

---

## 📌 Next Steps

1️⃣ **Test Hailo-8L AI acceleration** for LLM offloading.  
2️⃣ **Optimize embeddings for RAG** (ensure consistent vector sizes).  
3️⃣ **Compare FAISS vs. ChromaDB** for vector search.  
4️⃣ **Document & automate** the setup into scripts for easy deployment.

---

## 📜 Logs & Documentation

- **System logs:** `rpi5_ai_setup_log.txt`  
- **LLM Context Length Tests:** `mistral_ctx_test.log`  
- **Benchmark results:** `benchmark_results.log`  
- **Hailo Logs:** `hailort.log`  

---

## 🔐 Security & Secrets

This project now **excludes** all sensitive files (like OAuth credentials or service account JSONs) from version control. These files are stored in a secure folder outside the repo (`~/secure-keys/`). To ensure you don’t accidentally commit them:

1. **Add** credential filenames to `.gitignore` (e.g. `client_secret.json`, `api-credentials.json`, `service_account_key.json`).  
2. Never commit or push these files to GitHub.  
3. If you accidentally commit them, **revoke** and **regenerate** the keys in [Google Cloud Console](https://console.cloud.google.com/apis/credentials).

---

## 📂 Google Drive Integration

We use a **service account** to fetch files from Google Drive headlessly:

1. **Service Account & Key**: Create a service account in Google Cloud, grant it access to your Drive folder, and store its JSON key in `~/secure-keys/`.  
2. **Download** or **list** files with the Drive API in Python.  
3. **Parse** and **embed** the text, then index in FAISS.

This ensures no interactive OAuth is needed on the Pi.

---

## 🤝 Contributing

Feel free to submit issues & PRs to improve this **Raspberry Pi AI setup!**
This project is **open-source**, and contributions are welcome. 🚀

---

## 📌 References
- [Hailo AI Hat Docs](https://hailo.ai/)
- [Raspberry Pi 5 Official Docs](https://www.raspberrypi.com/)
- [FAISS Vector Search](https://faiss.ai/)
- [Mistral AI Models](https://mistral.ai/)

---

## Retrospective Reflection on the Raspberry Pi 5 AI Project 🚀

1️⃣ **What Went Well?** ✅  
- **Hands-on Learning**: You’ve successfully set up Mistral-7B (Q4_K_M) on Raspberry Pi 5, tested different configurations, and integrated FAISS for vector search.  
- **AI Hat Integration**: The Hailo-8L accelerator is installed, recognized, and responding to system queries.  
- **Performance & Stability Testing**: You stress-tested `n_ctx` values, monitored RAM/CPU usage/temperatures, and used zswap to improve memory management.  
- **Documentation & GitHub**: Well-structured commits and progress logs make the project easier to replicate for others.

2️⃣ **What Was Challenging?** 🤔  
- **Debugging Loops**: Issues around FAISS embeddings, model inconsistencies, and Hailo firmware led to repeated troubleshooting.  
- **Hailo Acceleration**: While recognized, full offloading hasn’t been confirmed yet—some firmware warnings remain.  
- **LLM & RAG Confusion**: Mistral-7B is used for both chatbot and summarization RAG tasks, which might need separate configs or models.

3️⃣ **What Can Be Improved?** 📈  
- **Refactor Codebase**: Ensure embedding dimensions match for FAISS, confirm RAG pipelines are properly integrated.  
- **Clarify AI Acceleration**: More benchmarking is needed to confirm which workloads run on Hailo vs. CPU.  
- **Optimize Model Selection**: Possibly use a smaller, specialized model for RAG vs. a bigger model for chat.  
- **Smoother Setup**: A step-by-step guide or shell script would help new users replicate the environment.

4️⃣ **Next Steps** 🔜  
- **Confirm FAISS embeddings** are consistent and debug dimensional mismatch.  
- **Benchmark** and verify Hailo offloading vs. CPU inference.  
- **Differentiate RAG & Chatbot** usage (pick appropriate model sizes).  
- **Polish & Document** the final steps for public release.

Overall, the project has come a long way. With a bit more optimization and clarity on Hailo offloading, it’ll be an impressive local AI system running on the Pi 5! 🌍🚀
