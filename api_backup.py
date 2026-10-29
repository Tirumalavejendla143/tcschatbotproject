
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables (OpenAI key is no longer required)
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

# Connect the frontend to the backend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://127.0.0.1:8000",
            "http://localhost:8000"
        ]
    }
})

# Embedding model used when creating the FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Existing FAISS database
FAISS_DIR = BASE_DIR / "tcs_doc_index"

if not FAISS_DIR.is_dir():
    raise FileNotFoundError(
        f"FAISS index not found at: {FAISS_DIR}. "
        "Please check your existing index folder."
    )

print("Loading embedding model...", flush=True)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Loading FAISS vector database...", flush=True)

vector_store = FAISS.load_local(
    str(FAISS_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

print("FAISS database loaded successfully.", flush=True)

# Load the free local language model
MODEL_NAME = "google/flan-t5-small"

print("Loading FLAN-T5 tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading FLAN-T5 model...", flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

print("FLAN-T5 model loaded successfully.", flush=True)

# Prompt for answering questions from the PDF
prompt = ChatPromptTemplate.from_template("""
You are a helpful TCS document assistant.

Answer the question using only the provided PDF context.
Use simple and clear English.
Do not invent facts.
If the answer is not available in the context,
say that you could not find it in the document.

PDF context:
{context}

User question:
{question}

Answer:
""")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "TCS chatbot backend is running",
        "model": MODEL_NAME
    })

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
        # Retrieve relevant text from the existing PDF index
        documents = retriever.invoke(question)

        if not documents:
            return jsonify({
                "answer": (
                    "I could not find relevant information "
                    "in your TCS PDF."
                ),
                "sources": []
            })

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        # Prepare the prompt for FLAN-T5
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

        # Generate the answer locally
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=2,
                do_sample=False
            )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if not answer:
            answer = (
                "I could not generate an answer from "
                "the retrieved document context."
            )

        # Return source page numbers
        sources = []

        for doc in documents:
            sources.append({
                "page": doc.metadata.get("page"),
                "source": Path(
                    doc.metadata.get(
                        "source", "tcsreport.pdf"
                    )
                ).name
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

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
