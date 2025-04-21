import subprocess
import sys
import os

def setup_environment():
    print("Setting up environment for Satellite to Map Web Application...")
    
    # Check if virtual environment exists, create if it doesn't
    if not os.path.exists('.venv'):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', '.venv'])
    
    # Determine the pip executable to use
    if os.name == 'nt':  # Windows
        pip_executable = os.path.join('.venv', 'Scripts', 'pip')
    else:  # Unix/Linux/Mac
        pip_executable = os.path.join('.venv', 'bin', 'pip')
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([pip_executable, 'install', '--upgrade', 'pip'])
    
    # Install NumPy first to ensure compatibility
    print("Installing NumPy 1.26.4...")
    subprocess.check_call([pip_executable, 'install', 'numpy==1.26.4'])
    
    # Install other dependencies
    print("Installing dependencies...")
    subprocess.check_call([pip_executable, 'install', '-r', 'requirements.txt'])
    
    # Create necessary directories
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads', exist_ok=True)
    if not os.path.exists('static/generated'):
        os.makedirs('static/generated', exist_ok=True)
    
    print("\nSetup complete! You can now run the application with:")
    if os.name == 'nt':  # Windows
        print(".venv\\Scripts\\python app.py")
    else:  # Unix/Linux/Mac
        print(".venv/bin/python app.py")
    print("\nThen open your browser and navigate to: http://127.0.0.1:5000/")

if __name__ == '__main__':
    setup_environment() 