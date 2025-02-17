import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv
from services.answer_generation import generate_answer
from config.settings import MODELS
from services.vector_store import VectorStoreManager # Import VectorStoreManager


# Suppress Streamlit warnings
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables
def init_session_state():
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(MODELS.keys())[0]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager()

def handle_file_upload(uploaded_files):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"📂 Processing file: {uploaded_file.name}..."):
                    file_path = os.path.join('pdfs', uploaded_file.name)
                    os.makedirs('pdfs', exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        db = st.session_state.vector_store_manager.create_vector_store(file_path)
                        st.session_state.processed_files[uploaded_file.name] = db
                        if uploaded_file.name not in st.session_state.selected_files:
                            st.session_state.selected_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                    except ValueError as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            else:
                st.info(f"ℹ️ {uploaded_file.name} has already been processed.")

def handle_model_change():
    st.session_state.selected_model = st.session_state.temp_model

def handle_file_selection():
    st.session_state.selected_files = st.session_state.temp_selected_files

def clear_all_files():
    st.session_state.processed_files = {}
    st.session_state.selected_files = []
    st.session_state.chat_history = []

def submit_question():
    if st.session_state.question_input.strip():
        st.session_state.chat_history.append({"question": st.session_state.question_input})
        st.session_state.question_input = ""

def main():
    init_session_state()
    
    st.title("📄 PDF Question Answering App")

    # Model selection without triggering rerun
    st.selectbox(
        "Select a model:",
        options=list(MODELS.keys()),
        key="temp_model",
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        on_change=handle_model_change
    )

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader"
    )
    handle_file_upload(uploaded_files)

    # Display processed files and selection
    if st.session_state.processed_files:
        st.subheader("Available PDF Files:")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.multiselect(
                "Select PDFs to query:",
                options=list(st.session_state.processed_files.keys()),
                default=st.session_state.selected_files,
                key="temp_selected_files",
                on_change=handle_file_selection
            )
        
        with col2:
            if st.button("Clear All Files"):
                clear_all_files()
                st.rerun()

        # Example prompts
        st.markdown("""
        **Example questions you can ask:**
        - Summarize all the selected PDF documents together
        - What are the main findings across all papers?
        - Compare and contrast the methodologies used in these documents
        - What are the common themes or conclusions?
        """)

        # Chat history display
        for entry in st.session_state.chat_history:
            st.markdown(
                f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>"
                f"<b>User:</b> {entry['question']}</div>",
                unsafe_allow_html=True
            )
            if "answer" in entry:
                st.markdown(f"<b>Assistant:</b> {entry['answer']}", unsafe_allow_html=True)
            else:
                st.markdown("<b>Assistant:</b> Generating answer...", unsafe_allow_html=True)

        # Question input
        st.text_input(
            "🔍 Ask a question about the selected PDFs",
            key="question_input",
            on_change=submit_question,
            placeholder="e.g., 'Provide a comprehensive summary of all selected documents'"
        )

        # Answer generation
        if st.session_state.chat_history and "answer" not in st.session_state.chat_history[-1]:
            question = st.session_state.chat_history[-1]["question"]

            if not st.session_state.selected_files:
                st.warning("⚠️ Please select at least one PDF file.")
            else:
                selected_vector_stores = [
                    st.session_state.processed_files[filename]
                    for filename in st.session_state.selected_files
                ]
                combined_db = st.session_state.vector_store_manager.combine_vector_stores(selected_vector_stores)

                if combined_db is None:
                    st.warning("⚠️ No vector stores were combined.")
                else:
                    related_documents = st.session_state.vector_store_manager.similarity_search(combined_db, question, k=4)
                    context = "\n\n".join([doc.page_content for doc in related_documents])
                    try:
                        answer = generate_answer(question, context, st.session_state.selected_model)
                        if answer:
                            st.session_state.chat_history[-1]["answer"] = answer
                        else:
                            st.session_state.chat_history[-1]["answer"] = "❌ An error occurred while generating the answer."
                    except Exception as e:
                        logging.error(f"Unexpected error: {e}")
                        st.session_state.chat_history[-1]["answer"] = "❌ An unexpected error occurred."

                    st.rerun()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n🛑 Streamlit app interrupted. Exiting cleanly...")
        sys.exit(0)
