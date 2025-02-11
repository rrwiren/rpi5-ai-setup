# Raspberry Pi 5 AI Setup 🚀

This project focuses on deploying **local LLMs, vector search, and AI-enhanced workflows** on the **Raspberry Pi 5**.  
It includes optimizations for running **efficient** AI models, leveraging **Hailo-8L AI acceleration**, and integrating **vector search**.

---

## 🏗️ Project Overview

### ✅ Current AI Stack:
| **Component**        | **Model/Technology**           | **Purpose**                  |
|----------------------|--------------------------------|--------------------------------|
| **LLM (Chatbot)**   | Gemma-2B (Q4_K_M)             | General LLM responses         |
| **RAG (Summarization)** | Mistral-7B (Q4_K_M)         | Retrieval-augmented generation |
| **Vector Search**   | FAISS / ChromaDB              | Efficient search & retrieval  |
| **Hardware Acceleration** | Hailo-8L AI Hat         | Offloading AI tasks           |

---

## 🔧 Setup & Configuration

### 1️⃣ Raspberry Pi 5 Setup:
- Installed **Raspberry Pi OS (64-bit)**
- Set up SSH, locales, and expanded filesystem.
- Configured **Zswap & memory optimizations** (`zswap.enabled=1`, `vm.swappiness=100`).
- Optimized **power & thermal management**.

### 2️⃣ Local LLM Inference:
- Installed `llama-cpp-python` to run **Mistral-7B (Q4_K_M)**.
- Successfully tested **n_ctx=16384** as a stable max context window.

### 3️⃣ Vector Search:
- FAISS **1.10.0** installed and configured.
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
|-----------|------------------|-------------|-------------|
| **CPU**   | 45.79 sec        | 0.92 GB     | 75.7°C      |
| **Hailo** | 45.85 sec        | 0.92 GB     | 76.3°C      |

### **Vector Search (FAISS):**
| **Dataset**    | **Indexing Time** | **Query Speed** |
|---------------|------------------|----------------|
| Small Corpus | 3.2 sec           | 0.5 ms/query  |
| Large Corpus | TBD               | TBD           |

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
- **Hailo Logs:** `hailo.log`

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
