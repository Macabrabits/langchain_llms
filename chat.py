from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})
llm = Ollama(model="llama3:8b")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Consultar
# query = "Responda em português: Como o Karpenter lida com nodes spot?"
query = "Como o Karpenter lida com nodes spot?"
res = qa.run(query)
print(res)
