import subprocess
import time
import os

# Define log file
LOG_FILE = "hailo_diagnostics.log"

def run_command(command):
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception as e:
        return str(e)

def log_result(section, output):
    """Log output to file and print it."""
    log_entry = f"\n### {section} ###\n{output}\n{'='*40}"
    print(log_entry)
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_entry + "\n")

def check_hailo_status():
    """Check Hailo driver and firmware version."""
    log_result("Checking Hailo Firmware & Driver", run_command("hailortcli fw-info"))

def list_hailo_devices():
    """List available Hailo devices."""
    log_result("Listing Hailo Devices", run_command("hailortcli device-info"))

def run_hailo_benchmark():
    """Run Hailo AI benchmark."""
    log_result("Running Hailo Benchmark", run_command("hailortcli benchmark"))

def monitor_system_load():
    """Check CPU & Memory usage before/after Hailo benchmark."""
    pre_load = run_command("top -bn1 | head -n 20")
    log_result("System Load BEFORE Hailo Benchmark", pre_load)

    print("\nRunning Hailo Benchmark for 10 seconds...")
    time.sleep(10)  # Simulate inference test

    post_load = run_command("top -bn1 | head -n 20")
    log_result("System Load AFTER Hailo Benchmark", post_load)

def check_dmesg_logs():
    """Check system logs for Hailo-related messages."""
    log_result("Checking dmesg for Hailo Errors", run_command("dmesg | grep -i 'hailo'"))

def main():
    """Run all diagnostics."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)  # Clear previous logs

    print("🚀 Running Hailo-8L Diagnostics...\n")
    check_hailo_status()
    list_hailo_devices()
    run_hailo_benchmark()
    monitor_system_load()
    check_dmesg_logs()

    print(f"\n✅ Diagnostics complete! Log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
