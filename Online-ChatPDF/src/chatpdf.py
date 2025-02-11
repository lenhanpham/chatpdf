import streamlit as st
import os
import logging
from dotenv import load_dotenv
from utils.file_processing import create_vector_store
from services.answer_generation import generate_answer
from config.settings import MODELS

# Suppress Streamlit warnings
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# Streamlit UI
if __name__ == "__main__":
    try:
        st.title("📄 PDF Question Answering App")

        selected_model = st.selectbox(
            "Select a model:",
            options=list(MODELS.keys()),
            index=0
        )

        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"📂 Processing file: {uploaded_file.name}..."):
                    file_path = os.path.join('pdfs', uploaded_file.name)
                    os.makedirs('pdfs', exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        db = create_vector_store(file_path)
                        st.session_state.processed_files[uploaded_file.name] = db
                        st.success("✅ PDF processed successfully! You can now ask questions.")
                    except ValueError as e:
                        st.error(f"❌ Error processing PDF: {e}")
            else:
                st.info("ℹ️ This PDF has already been processed. Using cached data.")

            st.session_state.db = st.session_state.processed_files[uploaded_file.name]

        for entry in st.session_state.chat_history:
            st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>"
                        f"<b>User:</b> {entry['question']}</div>", unsafe_allow_html=True)
            if "answer" in entry:
                st.markdown(f"<b>Assistant:</b> {entry['answer']}", unsafe_allow_html=True)
            else:
                st.markdown("<b>Assistant:</b> Generating answer...", unsafe_allow_html=True)

        def submit_question():
            st.session_state.chat_history.append({"question": st.session_state.question_input})
            st.session_state.question_input = ""

        st.text_input(
            "🔍 Ask a question about the uploaded PDF",
            key="question_input",
            on_change=submit_question
        )

        if st.session_state.chat_history and "answer" not in st.session_state.chat_history[-1]:
            question = st.session_state.chat_history[-1]["question"]
            if st.session_state.db is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                related_documents = st.session_state.db.similarity_search(question, k=4)
                context = "\n\n".join([doc.page_content for doc in related_documents])
                try:
                    answer = generate_answer(question, context, selected_model)
                    if answer:
                        st.session_state.chat_history[-1]["answer"] = answer
                    else:
                        st.session_state.chat_history[-1]["answer"] = "❌ An error occurred while generating the answer."
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    st.session_state.chat_history[-1]["answer"] = "❌ An unexpected error occurred."

                st.rerun()

    except KeyboardInterrupt:
        logging.info("\n🛑 Streamlit app interrupted. Exiting cleanly...")
        sys.exit(0)