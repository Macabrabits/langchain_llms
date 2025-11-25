# from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
import dotenv

dotenv.load_dotenv()

llm = ChatOllama(
    model="llama3:8b",
    temperature=0,
    # other params...
)

messages = [
    (
        "system",
        "Be technical and direct in your answers.",
    ),
    # ("human", "I love programming."),
    ("human", "How does karpenter deal with spot nodes?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
