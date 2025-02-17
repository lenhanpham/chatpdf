from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

class VectorStoreManager:
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    def create_vector_store(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)

        try:
            db = FAISS.from_documents(chunked_docs, self.embeddings_model)
            return db
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise ValueError("Failed to create vector store. Please check the embedding model.")

    def combine_vector_stores(self, vector_stores):
        """Combines multiple FAISS vector stores into one."""
        if not vector_stores:
            return None

        combined_db = vector_stores[0]
        for db in vector_stores[1:]:
            combined_db.index.merge_from(db.index)
        return combined_db

    def similarity_search(self, db, query, k=8):
        if db is None:
            raise ValueError("Vector store is not initialized.")
        return db.similarity_search(query, k=k)