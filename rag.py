import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HospitalRAG:
    def __init__(self, persist_directory: Path, llm, chunk_size=500, chunk_overlap=100):
        self.persist_directory = persist_directory
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _compute_hash(self, file_bytes):
        return hashlib.sha256(file_bytes).hexdigest()

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def load_document_store(self, file_bytes, file_name):
        collection_name = f"hospital_{self._compute_hash(file_bytes)}"

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        upload_path = self.persist_directory / file_name
        upload_path.write_bytes(file_bytes)

        documents = self._load_pdf(upload_path)
        chunks = self._split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(self.persist_directory),
        )

        return vectorstore

    def retrieve_context(self, vectorstore, question, top_k=3):
        docs = vectorstore.similarity_search(question, k=top_k)
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_question(self, vectorstore, question):
        context = self.retrieve_context(vectorstore, question)

        if not context.strip():
            return "Sorry, I couldn't find that information in the hospital document."

        prompt = f"""
You are a Hospital FAQ Assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:

Sorry, I couldn't find that information in the hospital document.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content