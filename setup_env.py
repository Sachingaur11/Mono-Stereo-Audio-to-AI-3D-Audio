import os
import subprocess
import sys

# Define the required packages
required_packages = [
    "pydub",
    "pymongo",
    "gridfs",
    "soundfile",
    "numpy",
    "requests",
    "scipy",
    "flask",
    "werkzeug"
]

# Step 1: Create a virtual environment
def create_virtual_env(env_name="venv"):
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", env_name])

# Step 2: Install the required packages
def install_packages(env_name="venv"):
    print("Installing packages...")
    # Activate virtual environment and install packages
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(env_name, "Scripts", "pip")
    else:  # macOS/Linux
        pip_path = os.path.join(env_name, "bin", "pip")
    
    subprocess.run([pip_path, "install", "--upgrade", "pip"])  # Upgrade pip
    subprocess.run([pip_path, "install"] + required_packages)

# Main function
if __name__ == "__main__":
    env_name = "venv"
    create_virtual_env(env_name)
    install_packages(env_name)
    print(f"Virtual environment '{env_name}' created and packages installed.")
