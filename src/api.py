from fastapi import FastAPI
from src.query import ask_question

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/ask")
def ask(q: str):
    answer, _ = ask_question(q)
    return {"answer": answer}