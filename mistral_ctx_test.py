 #!/usr/bin/env python3
# Created by Richard Wire together with AI :-)n
import os
import psutil
import time
from llama_cpp import Llama

# Path to your model
MODEL_PATH = "/home/riku/models/mistral-7b-v0.1.Q4_K_M.gguf"

# Context sizes to test
N_CTX_VALUES = [8192, 12288, 16384] # This seems to be optimum without crashing little rpi


# Thresholds for safety (adjust as needed)
MAX_MEMORY_USAGE_GB = 14.5  # Keep a buffer to avoid swap thrashing
LOG_FILE = "mistral_ctx_test.log"

# Simple test prompt
PROMPT = "Explain the significance of the Raspberry Pi 5 in AI development."

def log_result(n_ctx, load_time, eval_time, memory_used, response):
    """Logs test results to a file and prints to console."""
    log_entry = f"""
✅ n_ctx={n_ctx}
  Model Load Time: {load_time:.2f} sec
  Query Eval Time: {eval_time:.2f} sec
  Memory Used: {memory_used:.2f} GB
  Response: {response[:100]}...
"""
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

def get_memory_usage():
    """Returns current memory usage in GB."""
    return psutil.virtual_memory().used / (1024 ** 3)

def test_n_ctx(n_ctx):
    """Loads the model with specified context and runs a test query."""
    print(f"\n🚀 Testing n_ctx={n_ctx}...")
    start_mem = get_memory_usage()

    try:
        # Load model with the given context length
        start_time = time.time()
        llm = Llama(model_path=MODEL_PATH, n_ctx=n_ctx)
        load_time = time.time() - start_time

        # Run test query
        start_time = time.time()
        response = llm(PROMPT)
        eval_time = time.time() - start_time

        # Get memory usage
        memory_used = get_memory_usage() - start_mem

        # Log results
        log_result(n_ctx, load_time, eval_time, memory_used, response["choices"][0]["text"])

        return memory_used < MAX_MEMORY_USAGE_GB

    except Exception as e:
        print(f"❌ n_ctx={n_ctx} failed: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"\n🔍 Starting context size tests. Logs saved in {LOG_FILE}\n")
    with open(LOG_FILE, "w") as f:
        f.write("🔍 Mistral Context Length Test Results\n")

    for n_ctx in N_CTX_VALUES:
        if not test_n_ctx(n_ctx):
            print(f"\n⚠️ Memory limit reached at n_ctx={n_ctx}. Stopping tests.")
            break

    print(f"\n📄 Test results saved in: {LOG_FILE}")
