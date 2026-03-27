from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
db = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 10})

# LLM (use mistral if possible)
llm = Ollama(model="tinyllama",temperature=0)  # change to tinyllama if needed


def ask_question(query: str):
    try:
        docs = retriever.invoke(query)
        # 🔍 DEBUG START
        print("\n--- RETRIEVED DOCS ---")
        for d in docs:
            print(d.page_content[:200])
# 🔍 DEBUG END

        if not docs:
            return {"answer": "Not found in document"}

        # Combine context
        context = "\n".join([doc.page_content for doc in docs])

        # STRICT PROMPT
        prompt = f"""
You are an extraction system.

Rules:
- Answer in ONE short sentence
- Do NOT explain
- Do NOT generate examples
- Do NOT repeat the question
- Use ONLY the context
- If answer not found, say exactly: Not found in document

Context:
{context}

Question:
{query}

Answer:
"""

        # LLM call
        response = llm.invoke(prompt)

        # Extract text
        if hasattr(response, "content"):
            answer = response.content.strip()
        else:
            answer = str(response).strip()
            # ✅ CLEAN OUTPUT (ONLY THIS)
        answer = answer.split(".")[0].strip()

        if len(answer) < 10:
            return {"answer": "Not found in document"}

        return {"answer": answer + "."}

        # Remove garbage outputs
        bad_words = ["example", "question:", "answer:", "rule", "context"]
        for word in bad_words:
            if word in answer.lower():
                return {"answer": "Not found in document"}

        # Keep only first sentence
        answer = answer.split(".")[0].strip() + "."

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}