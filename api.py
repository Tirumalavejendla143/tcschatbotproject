
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://127.0.0.1:8000",
            "http://localhost:8000"
        ]
    }
})

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "google/flan-t5-small"
FAISS_DIR = BASE_DIR / "tcs_doc_index"

if not FAISS_DIR.is_dir():
    raise FileNotFoundError(
        f"Existing FAISS index not found: {FAISS_DIR}"
    )


# --------------------------------------------------
# 2. LOAD THE EXISTING FAISS DATABASE
# --------------------------------------------------

print("Loading embedding model...", flush=True)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Loading existing FAISS database...", flush=True)

vector_store = FAISS.load_local(
    str(FAISS_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

# Retrieve one potentially useful passage.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

print("FAISS database loaded successfully.", flush=True)


# --------------------------------------------------
# 3. LOAD THE LOCAL LANGUAGE MODEL
# --------------------------------------------------

print("Loading FLAN-T5 tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading FLAN-T5 model...", flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

print("FLAN-T5 model loaded successfully.", flush=True)


# --------------------------------------------------
# 4. QUESTION-ANSWERING PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You answer questions using only the provided TCS document passages.

Instructions:
1. Answer the exact question asked.
2. Find the specific person, date, number, or fact that answers it.
3. Do not answer with a company name when the question asks for a person's name.
4. Do not confuse Tata Group companies or historical job titles.
5. Never guess or use outside knowledge.
6. If the answer is not clearly stated in the passages, reply:
I could not find this information in the TCS document.

Document passages:
{context}

Question:
{question}

Give only the Answer:
""")


# --------------------------------------------------
# 5. HEALTH CHECK
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "TCS chatbot backend is running",
        "model": MODEL_NAME
    })


# --------------------------------------------------
# 6. CHAT API
# --------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Please enter a valid question."
        }), 400

    question = question.strip()

    try:
        # Search the existing PDF index.
        documents = retriever.invoke(question)

        if not documents:
            return jsonify({
                "answer": (
                    "I could not find this information "
                    "in the TCS document."
                ),
                "sources": []
            })

        # Prepare passages with their page numbers.
        passages = []

        for doc in documents:
            page = doc.metadata.get("page", "Unknown")
            passages.append(
                f"PDF page {page}:\n{doc.page_content}"
            )

        full_context = "\n\n".join(passages)

        # Keep context within a manageable token budget.
        # This helps prevent the prompt from being truncated
        # before the question reaches the model.
        context_tokens = tokenizer(
            full_context,
            truncation=True,
            max_length=300,
            return_tensors="pt"
        )

        context = tokenizer.decode(
            context_tokens["input_ids"][0],
            skip_special_tokens=True
        )

        formatted_prompt = prompt.format(
            context=context,
            question=question
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate an answer locally.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=2,
                do_sample=False,
                repetition_penalty=1.1
            )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if not answer:
            answer = (
                "I could not find this information "
                "in the TCS document."
            )

        # Log the question and retrieved pages for debugging.
        app.logger.info("Question: %s", question)
        app.logger.info(
            "Retrieved pages: %s",
            [
                doc.metadata.get("page", "Unknown")
                for doc in documents
            ]
        )
        app.logger.info("Generated answer: %s", answer)

        # Return source information to the frontend.
        sources = []

        for doc in documents:
            source_path = doc.metadata.get(
                "source", "tcsreport.pdf"
            )

            sources.append({
                "page": doc.metadata.get("page"),
                "source": Path(source_path).name
            })

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception:
        app.logger.exception("Chat request failed")

        return jsonify({
            "error": (
                "Failed to generate an answer. "
                "Check the backend terminal for details."
            )
        }), 500


# --------------------------------------------------
# 7. START THE SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
