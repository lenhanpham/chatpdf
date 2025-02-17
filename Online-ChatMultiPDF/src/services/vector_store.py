from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import numpy as np
from functools import lru_cache
import os

class VectorStoreManager:
    def __init__(self, model_name="BAAI/bge-small-en"):
        self._model_name = model_name
        self._embeddings_model = HuggingFaceEmbeddings(model_name=model_name)  # Initialize embeddings model

    def create_vector_store(self, file_path):
        """Create a vector store from a PDF file"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add file name to metadata for each document
        for doc in documents:
            doc.metadata['file_name'] = os.path.basename(file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)

        try:
            db = FAISS.from_documents(chunked_docs, self._embeddings_model)
            return db
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise

    def combine_vector_stores(self, vector_stores):
        """Combine multiple vector stores while preserving source information"""
        if not vector_stores:
            return None
        
        combined_db = vector_stores[0]
        for db in vector_stores[1:]:
            combined_db.merge_from(db)
        return combined_db

    def similarity_search(self, db, query, k=8):
        """Perform similarity search ensuring results from all documents"""
        try:
            # Get more results to ensure coverage of all documents
            results = db.similarity_search(query, k=k)
            
            # Group results by file name
            grouped_results = {}
            for doc in results:
                file_name = doc.metadata.get('file_name', 'Unknown')
                if file_name not in grouped_results:
                    grouped_results[file_name] = []
                grouped_results[file_name].append(doc)
            
            # Take at least 2 results from each file
            balanced_results = []
            for file_docs in grouped_results.values():
                balanced_results.extend(file_docs[:2])
            
            return balanced_results
            
        except Exception as e:
            logging.error(f"Error during similarity search: {e}")
            raise







