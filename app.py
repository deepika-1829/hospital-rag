import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from llm import get_llm
from rag import HospitalRAG

# Load environment variables
load_dotenv()

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Hospital FAQ Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital FAQ Assistant")
st.write(
    "Upload a Hospital Information PDF and ask questions about it using RAG."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📄 Upload Hospital PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

# Create ChromaDB directory if it doesn't exist
persist_dir = Path("chroma_db")
persist_dir.mkdir(parents=True, exist_ok=True)

vectorstore = None
rag = None

# -----------------------------
# Process Uploaded PDF
# -----------------------------
if uploaded_file is not None:

    st.sidebar.success("✅ PDF uploaded successfully.")

    try:
        llm = get_llm()

        rag = HospitalRAG(
            persist_directory=persist_dir,
            llm=llm
        )

        # Build Vector Store
        vectorstore = rag.load_document_store(
            uploaded_file.read(),
            uploaded_file.name
        )

        st.sidebar.success("✅ Embeddings created successfully.")
        st.sidebar.success("✅ Vector Database Ready!")

    except Exception as e:
        st.error(f"❌ {e}")

# -----------------------------
# Question Section
# -----------------------------
st.markdown("---")

question = st.text_input(
    "Ask a Hospital-related Question",
    placeholder="Example: Is emergency available 24 hours?"
)

if st.button("Ask"):

    if uploaded_file is None:
        st.warning("Please upload a Hospital PDF first.")

    elif question.strip() == "":
        st.warning("Please enter a question.")

    elif rag is None or vectorstore is None:
        st.error("Vector database is not initialized.")

    else:

        with st.spinner("Searching the document..."):

            try:
                answer = rag.answer_question(
                    vectorstore,
                    question
                )

                st.subheader("Question")
                st.write(question)

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ {e}")

st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit • LangChain • ChromaDB • HuggingFace • Groq • LangSmith"
)