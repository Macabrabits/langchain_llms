# langchain_llms
A project for testing different llm integrations like MCP and RAG

## 📁 Project Structure

```
agent/
├── src/
│   └── agent/              # Main package (modules only)
│       └── __init__.py
├── scripts/                # Executable scripts
│   ├── build_index.py      # Build vector database
│   ├── chat.py            # Chat with RAG + agents
│   ├── chat_rag.py        # Chat with RAG (advanced)
│   ├── chat_simple.py     # Simple chat without RAG
│   ├── rag_only.py        # RAG retrieval only
│   └── crawl.py           # Web crawler for data collection
├── data/                   # Data directory
│   ├── markdowns/          # Downloaded markdown files
│   └── chroma_db/          # Vector database
├── pyproject.toml          # Dependencies & project config
├── uv.lock                 # Locked versions
├── .python-version         # Python version (3.11)
└── .env                    # Environment variables (not committed)
```

## 🚀 Setup with `uv`

### 1. Install `uv` (one time)
```powershell
pip install uv
```

### 2. Create virtual environment and install dependencies
```powershell
uv sync
```

### 3. Run scripts
```powershell
# Using uv run (recommended - auto venv)
uv run python scripts/build_index.py
uv run python scripts/chat.py
uv run python scripts/rag_only.py
uv run python scripts/chat_simple.py
uv run python scripts/crawl.py

# OR activate venv manually
.\.venv\Scripts\Activate.ps1
python scripts/build_index.py
```

### 4. Add new dependencies
```powershell
# Option 1: Quick add
uv add new-package

# Option 2: Edit pyproject.toml and sync
uv sync
```

### 5. Development setup
```powershell
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run ruff check .
```

## 📋 Important Files
- **`pyproject.toml`** - All dependencies (replaces requirements.txt)
- **`uv.lock`** - Locked dependency versions (auto-generated)
- **`src/agent/`** - Python package (modules only, no scripts)
- **`scripts/`** - Executable scripts
- **`data/`** - Markdowns and vector database

## 🔧 Environment Variables
Create `.env` file (based on `.env_example`):
```
OLLAMA_BASE_URL=http://localhost:11434
FIRECRAWL_API_KEY=your_key_here
```

## 📝 Notes
- Use `uv run` for all script execution
- Paths in scripts automatically resolve to project root
- All data stored in `data/` directory
- Keep `src/agent/` for reusable modules only


