import os
import sys
import torch
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    log("Starting diagnostic...")
    log(f"Process ID: {os.getpid()}")
    log(f"Python version: {sys.version}")
    
    log("Attempting to import torch...")
    import torch
    log(f"Torch version: {torch.__version__}")
    
    log("Checking CUDA available...")
    available = torch.cuda.is_available()
    log(f"CUDA available: {available}")
    
    if available:
        log(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        log("Attempting to allocate a small tensor on CUDA:0...")
        x = torch.ones(1, device='cuda:0')
        log(f"Allocation success: {x.item()}")
        
        if torch.cuda.device_count() > 1:
            log("Attempting to allocate a small tensor on CUDA:1...")
            y = torch.ones(1, device='cuda:1')
            log(f"Allocation success: {y.item()}")
            
        log("Diagnostic complete. GPU subsystem is responsive.")
    else:
        log("Diagnostic complete. CUDA is NOT available.")

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
    sys.exit(1)
