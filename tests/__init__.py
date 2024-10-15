import sys
import os

# Get the current working directory
current_directory = os.getcwd()

# Add the current directory to the Python path if it's not already there
if current_directory not in sys.path:
    sys.path.append(current_directory)