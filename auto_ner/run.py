"""run the steamlit app through this python file"""

import os
import subprocess
import argparse

def ensure_folders_exist(script_dir):
    images_path = os.path.join(script_dir, "images")
    saved_model_path = os.path.join(script_dir, "saved_models")

    # Create the 'images' directory if it doesn't exist
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Create the 'saved_model' directory if it doesn't exist
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501,
                        help="Port number for the Streamlit app")
    args = parser.parse_args()

    # Get absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure that required folders exist
    ensure_folders_exist(script_dir)

    # Construct paths for app.py, images, and saved_model directories
    app_path = os.path.join(script_dir, "app.py")

    # Run the Streamlit app defined at app_path
    cmd = ["python", "-m", "streamlit", "run", "--server.port", str(args.port), app_path]
    subprocess.call(cmd)

if __name__ == "__main__":
    run()
