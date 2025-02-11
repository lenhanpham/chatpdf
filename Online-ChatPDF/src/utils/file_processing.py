from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    try:
        db = FAISS.from_documents(chunked_docs, embeddings_model)
        return db
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise ValueError("Failed to create vector store. Please check the embedding model.")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()