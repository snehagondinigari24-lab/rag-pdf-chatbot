import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_PATH = "vectorstore/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "tinyllama"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = None
retriever = None

if os.path.exists(DB_PATH):
    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.25}
    )
else:
    print(f"Warning: {DB_PATH} not found. Run ingest.py first.")

llm = Ollama(model=OLLAMA_MODEL, temperature=0)

PROMPT = PromptTemplate.from_template("""
You are a strict document question-answering assistant.

Use ONLY the context below.
If the answer is not clearly present in the context, respond exactly:
Not found in document.

Rules:
- Give one short, direct answer.
- Do not guess.
- Do not use outside knowledge.
- If the query is a heading/title and it exists in context, provide the related information from context.
- If only the heading exists but no supporting content exists, say: Not found in document.

Context:
{context}

Question:
{question}

Answer:
""")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def keywords(text: str):
    stop_words = {
        "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
        "at", "to", "for", "from", "by", "with", "and", "or", "if", "then",
        "when", "where", "who", "why", "how", "does", "do", "did", "can",
        "could", "should", "would", "tell", "me", "about", "please", "policy"
    }
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def exact_phrase_in_docs(query: str, docs) -> bool:
    nq = normalize(query)
    for doc in docs:
        if nq in normalize(doc.page_content):
            return True
    return False


def keyword_overlap_score(query: str, docs) -> float:
    q_words = set(keywords(query))
    if not q_words:
        return 0.0
    context_words = set(keywords(" ".join(doc.page_content for doc in docs)))
    overlap = q_words.intersection(context_words)
    return len(overlap) / max(len(q_words), 1)


def build_context(docs, max_chars=2500):
    parts = []
    total = 0
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def clean_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = re.sub(r"\s+", " ", answer)
    bad = ["question:", "answer:", "context:", "rules:"]
    if not answer or any(x in answer.lower() for x in bad):
        return "Not found in document."
    if answer.lower() in {"not found", "not found in document"}:
        return "Not found in document."
    if not answer.endswith((".", "!", "?")):
        answer += "."
    return answer


def ask_question(query: str):
    if db is None or retriever is None:
        return {"answer": "Error: Vector database not initialized."}

    query = query.strip()
    if not query:
        return {"answer": "Not found in document."}

    try:
        docs = retriever.invoke(query)

        if not docs:
            return {"answer": "Not found in document."}

        phrase_found = exact_phrase_in_docs(query, docs)
        overlap = keyword_overlap_score(query, docs)

        if not phrase_found and overlap < 0.4:
            return {"answer": "Not found in document."}

        context = build_context(docs)
        if not context:
            return {"answer": "Not found in document."}

        response = llm.invoke(PROMPT.format(context=context, question=query))
        answer = clean_answer(response)

        if answer == "Not found in document." and phrase_found:
            return {
                "answer": "The document contains 'Employee Support and Wellness Policy', but the retrieved content is not sufficient to extract a reliable answer."
            }

        return {"answer": answer}

    except Exception as e:
        msg = str(e).lower()
        if "10061" in msg or "connection refused" in msg:
            return {"answer": "Error: Ollama server is not running. Please start Ollama."}
        return {"answer": f"Error: {str(e)}"}