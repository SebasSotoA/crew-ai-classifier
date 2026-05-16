# crew_ai_classifier

CrewAI pipeline that cleans data, engineers features, trains a classifier, evaluates it, and produces deployment artifacts under `ml-agents/output/`.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with a model such as `llama3.1:8b` (see `ml-agents/agents.py` for the configured model)

## Setup

```powershell
cd ml-agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Run

```powershell
cd ml-agents
python main.py
```

Generated files (`clean.csv`, reports, `model.pkl`, etc.) are written to `ml-agents/output/` and are not committed to Git.

## Layout

- `ml-agents/main.py` — entrypoint
- `ml-agents/agents.py`, `tasks.py`, `tools.py` — crew definition
- `ml-agents/data/titanic.csv` — sample dataset
