import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Ensure session state variables exist
if "db" not in st.session_state:
    st.session_state.db = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# Define function to create vector store
def create_vector_store(file_path):
    # Load and split the PDF into smaller chunks
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Reduce chunk size and overlap for faster processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for faster embedding
        chunk_overlap=150,  # Reduced overlap
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    # Use embeddings and create a vector store
    embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
    db = FAISS.from_documents(chunked_docs, embeddings)
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
st.title("📄 PDF Question Answering App")

# File uploader (Always visible)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Reserve space for the question input box using a placeholder
question_placeholder = st.empty()

# Process uploaded file only once
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.file_name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_uploaded = True
        # Show processing status
        with st.spinner(f"📂 Processing file: {uploaded_file.name}..."):
            # Save file temporarily
            file_path = os.path.join('pdfs', uploaded_file.name)
            if not os.path.exists('pdfs'):
                os.makedirs('pdfs')
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Create vector store and store in session state
            st.session_state.db = create_vector_store(file_path)
            st.success("✅ PDF processed successfully! You can now ask questions.")

# Always render the question input box
question = question_placeholder.text_input("🔍 Ask a question about the uploaded PDF")

# Answer the question if there is a query
if question:
    if st.session_state.db is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.write(f"**User question:** {question}")
        # Retrieve relevant documents
        related_documents = st.session_state.db.similarity_search(question, k=4)
        st.write(f"📚 Retrieved {len(related_documents)} relevant document sections.")
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in related_documents])
        # Generate answer using the model and context
        chain = prompt | model
        answer = chain.invoke({"question": question, "context": context})
        st.write(f"💡 **Answer:** {answer}")