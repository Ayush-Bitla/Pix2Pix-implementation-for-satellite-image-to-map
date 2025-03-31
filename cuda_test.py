import torch
import sys
import platform

def check_cuda():
    """Test CUDA availability and print detailed information"""
    print("=== PyTorch CUDA Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\nCUDA is NOT available. Possible reasons:")
        print("1. You don't have an NVIDIA GPU")
        print("2. NVIDIA drivers are not installed or outdated")
        print("3. You've installed a CPU-only version of PyTorch")
        print("\nTo fix:")
        print("- If you have an NVIDIA GPU, update your NVIDIA drivers")
        print("- Install the CUDA toolkit that's compatible with your PyTorch version")
        print("- Reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/")
        
        # Check if CUDA capabilities are compiled in PyTorch
        try:
            torch.ones(1).cuda()
            print("\nPyTorch was built with CUDA, but CUDA installation is not found.")
        except Exception as e:
            if "not compiled with CUDA" in str(e):
                print("\nYour PyTorch was built WITHOUT CUDA support.")
                print("You need to reinstall PyTorch with a CUDA-enabled version.")
            else:
                print(f"\nError occurred: {e}")
    else:
        # Get CUDA details
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # Get GPU details
        device_count = torch.cuda.device_count()
        print(f"\nNumber of CUDA devices: {device_count}")
        
        for i in range(device_count):
            print(f"\nDevice {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            
            # Test GPU computation
            x = torch.rand(1000, 1000, device=f'cuda:{i}')
            y = torch.rand(1000, 1000, device=f'cuda:{i}')
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            z = x @ y  # Matrix multiplication
            end_event.record()
            
            # Wait for the events to complete
            torch.cuda.synchronize()
            print(f"  Matrix multiplication time: {start_event.elapsed_time(end_event):.2f} ms")
            
            # Memory information
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            
        # Test a simple network forward pass
        print("\nTesting network forward pass...")
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10)
        ).cuda()
        
        input_tensor = torch.randn(16, 1000).cuda()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = model(input_tensor)
        end_event.record()
        
        torch.cuda.synchronize()
        print(f"Forward pass time: {start_event.elapsed_time(end_event):.2f} ms")
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    check_cuda() 