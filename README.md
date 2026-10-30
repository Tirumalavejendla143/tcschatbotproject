# TCS GenAI RAG Chatbot

An AI-powered chatbot that answers questions about TCS using information retrieved from a TCS PDF document.

## Project Overview

This project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a TCS PDF and generate answers using the FLAN-T5 language model.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- LangChain
- FAISS Vector Database
- Hugging Face Transformers
- Sentence Transformers
- FLAN-T5 Small

## Project Structure

```text
tcschatbotproject/
│
├── app.py
├── batch.py
├── tcs_doc_index/
├── documents/
│   └── tcsreport.pdf
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── README.md
└── .gitignore
```

## Features

- PDF-based question answering
- Semantic search using FAISS
- Text embeddings using all-MiniLM-L6-v2
- Answer generation using FLAN-T5 Small
- Flask REST API
- ChatGPT-style web interface
- Real-time question and answer interaction

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd tcschatbotproject
```

### 2. Create a virtual environment

```bash
python -m venv chatbotenv
```

### 3. Activate the environment (Windows)

```bash
chatbotenv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask backend

```bash
python app.py
```

Backend URL:

`http://127.0.0.1:5000`

### 6. Run the frontend

Open another terminal and run:

```bash
python -m http.server 8001 --directory frontend
```

Open in your browser:

`http://127.0.0.1:8001`

## API Endpoint

**POST** `/tcs`

Request:

```json
{
  "tcs_question": "Who is the chairman of TCS?"
}
```

Response:

```json
{
  "gpt_response": "Answer generated from the retrieved document."
}
```

## Models Used

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Language Model: `google/flan-t5-small`

## Author

Developed as a GenAI and Retrieval-Augmented Generation (RAG) project.
