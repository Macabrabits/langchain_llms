from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import dotenv

dotenv.load_dotenv()

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
llm = Ollama(model="llama3:8b")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Consultar
# query = "Responda em português: Como o Karpenter lida com nodes spot?"
query = "Como o Karpenter lida com nodes spot?"
res = qa.run(query)
print(res)
