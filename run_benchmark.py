import time
import torch
import psutil
import subprocess
import argparse
from llama_cpp import Llama

# Model Configuration
MODEL_PATH = "/home/riku/models/mistral-7b-v0.1.Q4_K_M.gguf"

# CLI Arguments
parser = argparse.ArgumentParser(description="Run AI Benchmark on RPi5")
parser.add_argument("--mode", choices=["cpu", "hailo"], default="cpu", help="Run on CPU or Hailo-8L AI Hat")
args = parser.parse_args()

# System Monitoring
def get_system_stats():
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout.strip()
    return {
        "RAM Used": f"{ram.used / 1e9:.2f} GB",
        "Swap Used": f"{swap.used / 1e6:.2f} MB",
        "CPU Temp": temp
    }

# Load Model
print(f"🔄 Loading model from: {MODEL_PATH} (Mode: {args.mode})")
#llm = Llama(model_path=MODEL_PATH, n_ctx=4096, embedding=True)

llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_batch=256, verbose=False)


# Perform Simple Query
query = "Explain the impact of AI on edge computing."
start_time = time.time()
response = llm.create_completion(prompt=query, max_tokens=100)
end_time = time.time()

# System Stats
stats = get_system_stats()
inference_time = end_time - start_time

print("\n✅ Benchmark Results")
print(f"🕒 Inference Time: {inference_time:.2f} sec")
for key, value in stats.items():
    print(f"🔹 {key}: {value}")

# Save Results
with open("benchmark_results.log", "a") as log:
    log.write(f"{args.mode.upper()} Mode - {time.ctime()}\n")
    log.write(f"Inference Time: {inference_time:.2f} sec\n")
    for key, value in stats.items():
        log.write(f"{key}: {value}\n")
    log.write("=" * 40 + "\n")

print("\n📄 Benchmark results saved in benchmark_results.log")
