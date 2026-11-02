import base64
import os
import subprocess
import sys
import tempfile
from crewai.tools import tool

WORKSPACE = "./output"
os.makedirs(WORKSPACE, exist_ok=True)

# Avoid flooding the LLM if someone materializes a huge frame.
_MAX_TOOL_CHARS = 120_000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TOOL_CHARS:
        return text
    return (
        text[: _MAX_TOOL_CHARS]
        + f"\n\n... (output truncated to {_MAX_TOOL_CHARS} characters)"
    )


def _runner_script_payload(user_code: str) -> str:
    """Python source run in a subprocess: exec user code, then summarize if silent."""
    b64 = base64.b64encode(user_code.encode("utf-8")).decode("ascii")
    return (
        "import base64\n"
        "import io\n"
        "import sys\n"
        "import traceback\n"
        "import types\n"
        "from contextlib import redirect_stdout, redirect_stderr\n"
        f"_code = base64.b64decode({b64!r}).decode('utf-8')\n"
        "_ns = {}\n"
        "_out = io.StringIO()\n"
        "_err = io.StringIO()\n"
        "try:\n"
        "    with redirect_stdout(_out), redirect_stderr(_err):\n"
        "        exec(compile(_code, '<user_code>', 'exec'), _ns, _ns)\n"
        "except Exception:\n"
        "    sys.stdout.write(traceback.format_exc())\n"
        "    sys.exit(1)\n"
        "_combined = _out.getvalue() + _err.getvalue()\n"
        "if _combined.strip():\n"
        "    sys.stdout.write(_combined)\n"
        "    sys.exit(0)\n"
        "lines = []\n"
        "try:\n"
        "    import pandas as pd\n"
        "except ImportError:\n"
        "    pd = None\n"
        "\n"
        "def _show(name, obj):\n"
        "    if pd is None:\n"
        "        return\n"
        "    if isinstance(obj, pd.DataFrame):\n"
        "        lines.append('== ' + name + ' (DataFrame, shape=' + str(obj.shape) + ') ==\\n'\n"
        "            + obj.head(15).to_string() + '\\n')\n"
        "    elif isinstance(obj, pd.Series):\n"
        "        lines.append('== ' + name + ' (Series, len=' + str(len(obj)) + ') ==\\n'\n"
        "            + obj.head(50).to_string() + '\\n')\n"
        "\n"
        "_priority = ('df', 'head', 'dtypes', 'describe', 'nulls', 'duplicates',\n"
        "             'clean', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test')\n"
        "for _n in _priority:\n"
        "    if _n in _ns:\n"
        "        _show(_n, _ns[_n])\n"
        "if pd is not None:\n"
        "    for _k in sorted(_ns.keys()):\n"
        "        if _k.startswith('_') or _k in _priority:\n"
        "            continue\n"
        "        _v = _ns[_k]\n"
        "        if isinstance(_v, types.ModuleType):\n"
        "            continue\n"
        "        if isinstance(_v, (pd.DataFrame, pd.Series)):\n"
        "            _show(_k, _v)\n"
        "if lines:\n"
        "    sys.stdout.write(\n"
        "        '[execute_python] No prints captured; auto-summary of DataFrame/Series:\\n\\n'\n"
        "        + ''.join(lines))\n"
        "else:\n"
        "    _scalar_lines = []\n"
        "    _scalar_types = (dict, list, tuple, set, int, float, bool, str)\n"
        "    try:\n"
        "        import numpy as _np\n"
        "        _np_arr = _np.ndarray\n"
        "    except ImportError:\n"
        "        _np_arr = None\n"
        "    for _k in list(_priority) + sorted(k for k in _ns if k not in _priority):\n"
        "        if _k not in _ns or _k.startswith('_'):\n"
        "            continue\n"
        "        _v = _ns[_k]\n"
        "        if isinstance(_v, types.ModuleType):\n"
        "            continue\n"
        "        if pd is not None and isinstance(_v, (pd.DataFrame, pd.Series)):\n"
        "            continue\n"
        "        _check = (_scalar_types,) if _np_arr is None else (_scalar_types, _np_arr)\n"
        "        if isinstance(_v, tuple(_check)):\n"
        "            _r = repr(_v)\n"
        "            if len(_r) > 2000:\n"
        "                _r = _r[:2000] + '... (truncated)'\n"
        "            _scalar_lines.append('== ' + _k + ' (' + type(_v).__name__ + ') ==\\n' + _r + '\\n')\n"
        "    if _scalar_lines:\n"
        "        sys.stdout.write('[execute_python] No prints; auto-repr of variables:\\n\\n'\n"
        "            + ''.join(_scalar_lines))\n"
        "    else:\n"
        "        sys.stdout.write('[execute_python] Finished with no output. Use print() to see results.\\n')\n"
    )


@tool("execute_python")
def execute_python(code: str) -> str:
    """
    ISOLATION RULE: Every call runs in a completely separate Python process.
    No variable, import, or DataFrame from any previous call exists here.
    You MUST include ALL imports AND ALL data-loading code at the top of EVERY
    single call — never assume anything from a prior call persists.

    Returns captured stdout + stderr. If nothing is printed, common pandas
    DataFrame/Series variables are auto-summarized, and dicts, numpy arrays,
    and scalars are shown via repr() so you always get feedback.
    Use print() explicitly for any value you want to see clearly.

    IMPORTANT: crewai tools (read_file, write_file, list_files) are not
    available inside this code; use open() for file I/O here.
    """
    script = _runner_script_payload(code)
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"\n[STDERR from interpreter]:\n{result.stderr}")
    if result.returncode != 0 and not parts:
        parts.append(f"(process exited with code {result.returncode})")

    text = "".join(parts).strip()
    if not text:
        text = (
            "[execute_python] No output and exit code "
            f"{result.returncode}. This should be rare; check code for early exit."
        )
    return _truncate(text)


@tool("read_file")
def read_file(path: str) -> str:
    """Reads and returns the contents of a file at the given path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the given path. Creates parent directories if needed."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool("list_files")
def list_files(directory: str) -> str:
    """Lists all files in a directory."""
    try:
        return "\n".join(os.listdir(directory))
    except Exception as e:
        return f"Error listing directory: {e}"
