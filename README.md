# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from a PDF document and get accurate answers.

## 🚀 Features
- Ask questions from PDF
- Uses FAISS for similarity search
- Powered by LLM (Mistral / Ollama)
- Simple Streamlit UI

## 🛠️ Tech Stack
- Python
- LangChain
- FAISS
- Streamlit
- Ollama

## 📂 Project Structure
src/
 ├── ingest.py
 ├── query.py
 ├── api.py
 └── streamlit_app.py

data/
 └── policy.pdf

## ▶️ Run Locally

pip install -r requirements.txt  
streamlit run src/streamlit_app.py

## 📌 Example Questions
- What is sick leave policy?
- What is resignation process?

## 🎯 Goal
Build a RAG chatbot for document-based Q&A.
