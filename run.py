"""run the steamlit app through this python file"""

# impiort the os module
import os
import subprocess
import argparse
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501,
                        help="Port number for the Streamlit app")
    args = parser.parse_args()


    """run the steamlit app"""
    # get absolute path to the app.py file
    app_path = os.path.dirname(os.path.abspath(__file__))
    #add app.py to the path with appropriate slashes for the os
    if os.name == "nt":
        app_path = app_path + "\\app.py"
    else:
        app_path = app_path + "/app.py"

    print(app_path)
    #run the streamlit app defined at app_path
    cmd = ["python", "-m", "streamlit", "run", "--server.port", str(args.port), app_path]
    subprocess.call(cmd)

if __name__ == "__main__":
    run()