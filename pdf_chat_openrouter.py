import streamlit as st
import os
import logging
import requests  # Ensure this import is included
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress Streamlit warnings
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # Cache processed files and their vector stores
if "db" not in st.session_state:
    st.session_state.db = None  # Store the current vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat history
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False  # Control input box reset

# Function to create vector store efficiently
def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Optimize chunk size to improve embedding performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Smaller chunks for faster processing
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    # Use HuggingFaceEmbeddings for generating embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    try:
        db = FAISS.from_documents(chunked_docs, embeddings_model)  # Generate embeddings and store in FAISS
        return db
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        raise ValueError("Failed to create vector store. Please check the embedding model.")

# Define the prompt template
template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. 
If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# Function to generate answers using OpenRouter API
MAX_CONTEXT_LENGTH = 4096  # Adjust based on the model's token limit

def generate_answer(question, context):
    # Truncate context if it exceeds the maximum length
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]
    
    data = {
        "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"},
        ]
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY_GEMINI_FL_2')}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(f"{os.getenv('OPENROUTER_BASE_URL')}/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the response JSON
        response_data = response.json()
        if not response_data or "choices" not in response_data or len(response_data["choices"]) == 0:
            logging.error("Invalid API response: Missing 'choices' key or empty list.")
            return None

        # Extract the answer from the response
        first_choice = response_data["choices"][0]
        if "message" not in first_choice or "content" not in first_choice["message"]:
            logging.error("Invalid API response: Missing 'message' or 'content' key.")
            return None

        answer = first_choice["message"]["content"]
        return answer

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while generating answer: {e}")
        return None

# Streamlit UI
if __name__ == "__main__":
    try:
        st.title("📄 PDF Question Answering App")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        # Handle file upload
        if uploaded_file is not None:
            # Check if the file has already been processed
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"📂 Processing file: {uploaded_file.name}..."):
                    file_path = os.path.join('pdfs', uploaded_file.name)
                    os.makedirs('pdfs', exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        # Create the vector store and cache it
                        db = create_vector_store(file_path)
                        st.session_state.processed_files[uploaded_file.name] = db
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                    except ValueError as e:
                        st.error(f"❌ Error processing PDF: {e}")
            else:
                # Use the cached vector store
                st.info("ℹ️ This PDF has already been processed. Using cached data.")

            # Set the current vector store
            st.session_state.db = st.session_state.processed_files[uploaded_file.name]

        # Display chat history
        for entry in st.session_state.chat_history:
            st.write(f"**User:** {entry['question']}")
            st.write(f"**Assistant:** {entry['answer']}")

        # Question input box
        question = st.text_input("🔍 Ask a question about the uploaded PDF", key="question_input")

        # Handle question submission
        if question and not st.session_state.clear_input:
            if st.session_state.db is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                # Save the question to chat history
                st.session_state.chat_history.append({"question": question})  # Save the question
                related_documents = st.session_state.db.similarity_search(question, k=4)
                context = "\n\n".join([doc.page_content for doc in related_documents])
                try:
                    answer = generate_answer(question, context)
                    if answer:
                        st.session_state.chat_history[-1]["answer"] = answer  # Save the answer
                    else:
                        st.session_state.chat_history[-1]["answer"] = "❌ An error occurred while generating the answer."
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    st.session_state.chat_history[-1]["answer"] = "❌ An unexpected error occurred."

                # Clear the question input box after processing
                st.session_state.clear_input = True  # Trigger input reset
                st.rerun()  # Trigger a rerun to update the UI

        # Reset the clear_input flag after clearing the input box
        if st.session_state.clear_input:
            st.session_state.clear_input = False

    except KeyboardInterrupt:
        logging.info("\n🛑 Streamlit app interrupted. Exiting cleanly...")
        sys.exit(0)