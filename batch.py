from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# step 1: load the pdf file
file_path = os.path.join("documents", "tcsreport.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

print("Total documents:", len(documents))

#step2: split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print("Total chunks:", len(chunks))

#step3:create a embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#step4 : create a faiss db
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

#step5: save the vector store to db permanently
vector_store.save_local("tcs_doc_index")
print("vector stored successfully")
