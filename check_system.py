import torch
import psutil
import os

def check_system():
    print("=== System Diagnostics ===")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1e9:.1f} GB")
    print(f"Available RAM: {memory.available / 1e9:.1f} GB")
    print(f"RAM usage: {memory.percent}%")
    
    # Check disk space
    disk = psutil.disk_usage('.')
    print(f"Disk space: {disk.free / 1e9:.1f} GB free")
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")

if __name__ == "__main__":
    check_system() 