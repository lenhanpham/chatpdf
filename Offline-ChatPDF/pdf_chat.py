import streamlit as st
import os
import sys
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Suppress Streamlit warnings
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to generate embeddings sequentially
def embed_chunk(chunk):
    embeddings_model = OllamaEmbeddings(model="deepseek-r1:7b")
    try:
        # Generate embedding without multiprocessing
        embedding = embeddings_model.embed_query(chunk.page_content)
        return embedding
    except Exception as e:
        logging.error(f"Error embedding chunk: {e}")
        return None

def embed_chunks(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}...")
        embedding = embed_chunk(chunk)
        if embedding:
            embeddings.append(embedding)
        else:
            logging.warning(f"Skipping chunk {i + 1} due to error.")
    return embeddings

# Function to create vector store efficiently
def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Optimize chunk size to improve embedding performance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300, add_start_index=True)
    chunked_docs = text_splitter.split_documents(documents)
    embeddings = embed_chunks(chunked_docs)
    if not embeddings:
        raise ValueError("No embeddings were generated. Please check the Ollama service and try again.")
    db = FAISS.from_embeddings(
        text_embeddings=list(zip([doc.page_content for doc in chunked_docs], embeddings)),
        embedding=OllamaEmbeddings(model="deepseek-r1:7b")
    )
    return db

# Define the prompt template
template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# Initialize the model for generating answers
model = OllamaLLM(model="deepseek-r1:7b")
prompt = ChatPromptTemplate.from_template(template)

# Streamlit UI
if __name__ == "__main__":
    try:
        if "db" not in st.session_state:
            st.session_state.db = None
        if "file_uploaded" not in st.session_state:
            st.session_state.file_uploaded = False
        if "file_name" not in st.session_state:
            st.session_state.file_name = None

        st.title("📄 PDF Question Answering App")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        question_placeholder = st.empty()

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.file_name:
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_uploaded = True
                with st.spinner(f"📂 Processing file: {uploaded_file.name}..."):
                    file_path = os.path.join('pdfs', uploaded_file.name)
                    os.makedirs('pdfs', exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        st.session_state.db = create_vector_store(file_path)
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                    except ValueError as e:
                        st.error(f"❌ Error processing PDF: {e}")

        question = question_placeholder.text_input("🔍 Ask a question about the uploaded PDF")
        if question:
            if st.session_state.db is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                st.write(f"**User question:** {question}")
                related_documents = st.session_state.db.similarity_search(question, k=4)
                st.write(f"📚 Retrieved {len(related_documents)} relevant document sections.")
                context = "\n\n".join([doc.page_content for doc in related_documents])
                chain = prompt | model
                try:
                    answer = chain.invoke({"question": question, "context": context})
                    st.write(f"💡 **Answer:** {answer}")
                except Exception as e:
                    logging.error(f"Error generating answer: {e}")
                    st.error("❌ An error occurred while generating the answer. Please try again.")

    except KeyboardInterrupt:
        logging.info("\n🛑 Streamlit app interrupted. Exiting cleanly...")
        sys.exit(0)