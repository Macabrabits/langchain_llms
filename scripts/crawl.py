# pip install firecrawl-py
import dotenv
from firecrawl import Firecrawl
import os
dotenv.load_dotenv()

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MARKDOWNS_DIR = os.path.join(DATA_DIR, "markdowns")

firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
os.makedirs(MARKDOWNS_DIR, exist_ok=True)

url = "https://karpenter.sh/docs"

# Scrape a website:
response = firecrawl.crawl(url,
    limit=40,
    scrape_options={
        'formats': [
            'markdown',
        ],
        'proxy': 'auto',
        'maxAge': 600000,
        'onlyMainContent': True
    }
)

print(len(response.data))

for i, item in enumerate(response.data, start=1):
    markdown_text = item.markdown
    filename = os.path.join(MARKDOWNS_DIR, f"{i}.md")
    with open(filename, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_text)
    print(f"Criado: {filename}")
