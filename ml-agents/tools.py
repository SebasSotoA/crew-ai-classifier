import os
import subprocess
import sys
import textwrap
from crewai.tools import tool

WORKSPACE = "./output"
os.makedirs(WORKSPACE, exist_ok=True)

@tool("execute_python")
def execute_python(code: str) -> str:
    """
    Executes arbitrary Python code in a subprocess.
    Returns stdout and stderr. Use this to inspect data,
    run transformations, train models, evaluate, etc.
    Always print() your results so they appear in the output.
    IMPORTANT: This runs in an isolated Python process — crewai tools such as
    read_file, write_file, and list_files are NOT available inside this code.
    Use open() to read or write files within execute_python code.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[STDERR]:\n{result.stderr}"
    return output or "(no output)"

@tool("read_file")
def read_file(path: str) -> str:
    """Reads and returns the contents of a file at the given path."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool("write_file")
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the given path."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool("list_files")
def list_files(directory: str) -> str:
    """Lists all files in a directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {e}"