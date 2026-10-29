# TCS GenAI PDF Chatbot 🤖

An AI-powered chatbot that answers questions based on TCS PDF documents using **Retrieval-Augmented Generation (RAG)**. The project combines document retrieval, vector search, and a language model to provide context-aware responses from the supplied documents.

## 🚀 Project Overview

The TCS GenAI PDF Chatbot allows users to ask questions about information contained in TCS documents. It retrieves relevant text from a preprocessed PDF knowledge base and uses that context to generate answers.

This project demonstrates the practical application of Generative AI, semantic search, vector databases, and Python-based API development.

## ✨ Features

- 📄 PDF document-based question answering
- 🔎 Semantic search using FAISS
- 🧠 Text embeddings using Sentence Transformers
- 💬 Context-aware answers using a language model
- 🐍 Python and Flask backend API
- 🌐 Web-based chatbot interface
- 🔗 REST API endpoints for health checks and chat requests

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Flask | Backend API |
| LangChain | RAG pipeline and document retrieval |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embeddings |
| Hugging Face Transformers | Language model integration |
| HTML, CSS, JavaScript | Frontend interface |
| Git and GitHub | Version control and source code management |

## 🏗️ Architecture

1. **Document loading:** Load the source PDF document.
2. **Text splitting:** Divide the document into smaller chunks.
3. **Embedding generation:** Convert text chunks into vector embeddings.
4. **Vector storage:** Store and retrieve embeddings using FAISS.
5. **Question processing:** Receive the user's question through the Flask API.
6. **Context retrieval:** Retrieve relevant document chunks.
7. **Answer generation:** Generate a response using the language model.
8. **Response display:** Return the answer to the chatbot interface.

## 📂 Project Structure

```text
tcschatbotproject/
├── documents/
│   └── tcsreport.pdf
├── tcs_doc_index/
├── templates/
├── api.py
├── api_backup.py
├── app.py
├── batch.py
├── test_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tirumalavejendla143/tcschatbotproject.git
cd tcschatbotproject
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the project

Check `api.py` for the configured embedding model, language model, and FAISS index path.

If your implementation requires environment variables, configure them locally in a `.env` file. Never commit API keys, passwords, or other secrets to GitHub.

### 5. Run the Flask backend

```bash
python api.py
```

Use the local address shown in the terminal. If your Flask application starts on port 5000, open:

`http://127.0.0.1:5000`

### 6. Test the API

If the health endpoint is enabled, visit:

`http://127.0.0.1:5000/api/health`

The chat endpoint is:

`POST /api/chat`

The exact request format depends on the implementation in `api.py`.

## 🧪 Example Use Cases

- Ask questions about information in the indexed TCS PDF.
- Retrieve relevant information from company documents.
- Explore how RAG combines document search with generative AI.
- Build a foundation for internal knowledge-base assistants.

## 🎯 Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG)
- Working with vector databases and embeddings
- Integrating Hugging Face models into Python applications
- Building REST APIs using Flask
- Connecting a frontend to an AI backend
- Managing and documenting projects with GitHub

## 🔮 Future Enhancements

- Support multiple PDF documents
- Improve response accuracy and source citations
- Add chat history and conversation management
- Improve the user interface and mobile responsiveness
- Deploy a public demo with suitable hosting and memory resources

## 👨‍💻 Author

**Tirumala Vejendla**

Electronics and Communication Engineering | Python | Generative AI | RAG | Machine Learning

## 🔗 Project Repository

[View the TCS GenAI PDF Chatbot on GitHub](https://github.com/Tirumalavejendla143/tcschatbotproject)

---

*This is an educational project demonstrating document-based question answering with Generative AI and Retrieval-Augmented Generation.*
