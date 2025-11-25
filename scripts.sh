pip install -r requirements.txt

uv run scripts/crawl.py
python scripts/build_index.py # Build the vector index from markdown files, not using uv, problem with cuda
python scripts/chat_rag.py
python scripts/chat_bedrock.py