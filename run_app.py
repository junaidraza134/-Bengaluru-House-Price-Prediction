import subprocess
import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Run streamlit
subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], cwd=script_dir)
