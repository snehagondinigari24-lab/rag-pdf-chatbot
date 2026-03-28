import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HF_TOKEN")
if token:
    print(f"✅ Token detected: {token[:5]}***")
else:
    print("❌ No token found in environment.")
def build_vector_db():
    # 1. Load PDF
    pdf_path = "data/policy.pdf" # Ensure this folder/file exists
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split Text (Mandatory for Unit Epsilon standards)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create and Save Index
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore/faiss_index")
    print("Vector database created successfully at vectorstore/faiss_index")

if __name__ == "__main__":
    build_vector_db()
