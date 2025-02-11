import streamlit as st
import os
import sys
import logging
import requests  # For OpenRouter API
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import Google Gemini API and Groq API
import google.generativeai as genai
from groq import Groq  # Import Groq client

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
if "question_input" not in st.session_state:
    st.session_state.question_input = ""  # Store the question input

# Define the available models and their corresponding API keys
MODELS = {
    "Google Gemini 2.0 Flash Lite (OpenRouter)": {
        "model_name": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_GEMINI_FL_2"),
        "type": "openrouter",
    },
    "Google Gemini 2.0 Pro (OpenRouter)": {
        "model_name": "google/gemini-2.0-pro-exp-02-05:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_GEMINI_PRO_2"),
        "type": "openrouter",
    },
    "Deepseek R1 Distill Llama 70B (OpenRouter)": {
        "model_name": "deepseek/deepseek-r1-distill-llama-70b:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_DS_R1LLAMA70B"),
        "type": "openrouter",
    },
    "Deepseek R1 (OpenRouter)": {
        "model_name": "deepseek/deepseek-r1:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_DS_R1"),
        "type": "openrouter",
    },
    "Google Gemini 2.0 Flash (Google API)": {
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_GEMINI_API_KEY"),
        "type": "google",
    },
    "Groq Llama 3.3 70B Versatile": {
        "model_name": "llama-3.3-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY"),
        "type": "groq",
    },
}

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

# Function to generate answers using OpenRouter, Google Gemini, or Groq API
MAX_CONTEXT_LENGTH = 4096  # Adjust based on the model's token limit

def generate_answer(question, context, selected_model):
    # Truncate context if it exceeds the maximum length
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    # Determine which API to use based on the selected model
    model_info = MODELS[selected_model]
    if model_info["type"] == "openrouter":
        # Use OpenRouter API
        data = {
            "model": model_info["model_name"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {question}\nContext: {context}"},
            ]
        }
        headers = {
            "Authorization": f"Bearer {model_info['api_key']}",
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

    elif model_info["type"] == "google":
        # Use Google Gemini API
        try:
            genai.configure(api_key=model_info["api_key"])
            model = genai.GenerativeModel(model_info["model_name"])
            response = model.generate_content(f"Question: {question}\nContext: {context}")
            return response.text
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None

    elif model_info["type"] == "groq":
        # Use Groq API
        try:
            client = Groq(api_key=model_info["api_key"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Question: {question}\nContext: {context}"},
                ],
                model=model_info["model_name"],
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            return None

# Streamlit UI
if __name__ == "__main__":
    try:
        st.title("📄 PDF Question Answering App")

        # Dropdown for selecting the model
        selected_model = st.selectbox(
            "Select a model:",
            options=list(MODELS.keys()),
            index=0  # Default to the first model
        )

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

        # Display chat history with colored user questions
        for entry in st.session_state.chat_history:
            # Display user question in light blue with bold "User:" label
            st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>"
                        f"<b>User:</b> {entry['question']}</div>", unsafe_allow_html=True)
            if "answer" in entry:
                # Display assistant answer in default text color with bold "Assistant:" label
                st.markdown(f"<b>Assistant:</b> {entry['answer']}", unsafe_allow_html=True)
            else:
                # Display placeholder while generating answer
                st.markdown("<b>Assistant:</b> Generating answer...", unsafe_allow_html=True)




        # Question input box with session state management
        def submit_question():
            """Submit the question and clear the input box."""
            st.session_state.chat_history.append({"question": st.session_state.question_input})  # Save the question
            st.session_state.question_input = ""  # Clear the input box

        st.text_input(
            "🔍 Ask a question about the uploaded PDF",
            key="question_input",
            on_change=submit_question
        )

        # Handle question submission
        if st.session_state.chat_history and "answer" not in st.session_state.chat_history[-1]:
            # If the last question doesn't have an answer yet, process it
            question = st.session_state.chat_history[-1]["question"]
            if st.session_state.db is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                related_documents = st.session_state.db.similarity_search(question, k=4)
                context = "\n\n".join([doc.page_content for doc in related_documents])
                try:
                    answer = generate_answer(question, context, selected_model)
                    if answer:
                        st.session_state.chat_history[-1]["answer"] = answer  # Save the answer
                    else:
                        st.session_state.chat_history[-1]["answer"] = "❌ An error occurred while generating the answer."
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    st.session_state.chat_history[-1]["answer"] = "❌ An unexpected error occurred."

                # Trigger a rerun to update the UI
                st.rerun()

    except KeyboardInterrupt:
        logging.info("\n🛑 Streamlit app interrupted. Exiting cleanly...")
        sys.exit(0)