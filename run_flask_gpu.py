#!/usr/bin/env python
"""
Startup script that ensures CUDA PyTorch is installed before running Flask app.
"""
import subprocess
import sys
from pathlib import Path

def ensure_cuda_torch():
    """Ensure CUDA-enabled PyTorch is installed."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available in current PyTorch installation")
            print("Installing CUDA-enabled PyTorch...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", 
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--upgrade"
            ], check=True)
            # Re-import to check
            import importlib
            importlib.reload(torch)
            if torch.cuda.is_available():
                print(f"✓ CUDA now available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                print("⚠ Still no CUDA after installation. Using CPU mode.")
                return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

if __name__ == '__main__':
    print("Checking GPU availability...")
    ensure_cuda_torch()
    
    print("\nStarting PDF Layout Extractor Flask App...")
    print("Open your browser to http://localhost:5000\n")
    
    from app import app
    # Disable reloader to avoid environment discrepancies in child process
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

