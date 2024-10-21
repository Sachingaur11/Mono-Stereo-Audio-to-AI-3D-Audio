import os
import subprocess
import sys

# Step 1: Create and activate a virtual environment
def create_and_activate_virtual_env(env_name="venv", python_version="python3.9"):
    print("Creating virtual environment...")
    subprocess.run([python_version, "-m", "venv", env_name])
    
    print("Activating virtual environment...")
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(env_name, "Scripts", "activate.bat")
    else:  # macOS/Linux
        activate_script = os.path.join(env_name, "bin", "activate")
    
    activate_command = f"source {activate_script}" if os.name != 'nt' else activate_script
    subprocess.run(activate_command, shell=True, executable="/bin/bash" if os.name != 'nt' else None)

# Main function
if __name__ == "__main__":
    env_name = "venv"
    create_and_activate_virtual_env(env_name)
    print(f"Virtual environment '{env_name}' created and activated using Python 3.9.")
