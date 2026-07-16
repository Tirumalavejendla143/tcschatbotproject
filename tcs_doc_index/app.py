from flask import Flask
from flask import request
from flask import jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app = Flask(__name__)

# Load embedding model once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database once
vector_store = FAISS.load_local(
    "tcs_doc_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load GPT-2 model once
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")



@app.route("/tcs", methods=["POST"])
def tcs_chatbot():

    # read a question
    data = request.get_json()
    question = data.get("tcs_question")

    # step 1: search relevant documents from FAISS
    context = vector_store.similarity_search(
        question,
        k=3
    )

    result = "\n".join(
        [doc.page_content for doc in context]
    )

    # step 2: create prompt
    prompt = f"""
You are a helpful TCS assistant.

Answer only based on the given context.
If the answer is not available in the context, say "I don't know."

Question:
{question}

Context:
{result}

Answer:
"""

    # step 3: pass prompt to GPT-2
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True, max_length=512)

    response = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
        
    )

    answer = tokenizer.decode(
        response[0],
        skip_special_tokens=True
    )

    return jsonify({
        "gpt_response": answer
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )