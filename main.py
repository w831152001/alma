from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox
import os

def read_directory_files(directory_path, subpath):   
    # Iterate through all files in the directory
    for filename in os.listdir(os.path.join(directory_path, subpath)):
        file_path = os.path.join(directory_path, subpath, filename)
        
        # Skip if it's a directory
        if os.path.isfile(file_path):
            # Read file contents in binary mode
            with open(file_path, "rb") as file:
                if subpath != "":
                    sbx.files.write(f"/alma/{subpath}/{filename}", file.read())
                else:
                    sbx.files.write(f"/alma/{filename}", file.read())

sbx = Sandbox() # Creates a persistent sandbox session
# sbx.files.mkdir("/alma") # Create a directory in the sandbox
read_directory_files("/workspace/alma", "")
read_directory_files("/workspace/alma", "core")
read_directory_files("/workspace/alma", "evals")
read_directory_files("/workspace/alma", "evals/agents")
read_directory_files("/workspace/alma", "evals/eval_envs")
read_directory_files("/workspace/alma", "evals/utils")
read_directory_files("/workspace/alma", "memo_archive")
read_directory_files("/workspace/alma", "memo_archive/baseline")
execution = sbx.run_code("""
import subprocess

subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd="/alma")
subprocess.run(["sed", "-i", "s/\\r$//", "training.sh"], cwd="/alma")
subprocess.run(
    ["sh", "training.sh"],
    cwd="/alma"
)
""") # Execute Python inside the sandbox
print(execution.logs)

files = sbx.files.list("/alma")
print(files)