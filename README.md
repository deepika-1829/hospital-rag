# Hospital FAQ Assistant using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) app that answers user questions from an uploaded hospital PDF.

## Features

- Upload a hospital PDF
- Load and split PDF text with LangChain Community
- Create embeddings with HuggingFace `all-MiniLM-L6-v2`
- Store vectors in ChromaDB under `chroma_db/`
- Retrieve top 3 relevant chunks by similarity search
- Answer questions using Groq Llama 3
- LangSmith tracing enabled for LLM and retrieval

## Files

- `app.py` - Streamlit UI and user interaction
- `rag.py` - Document loading, splitting, embedding, and retrieval
- `llm.py` - Groq LLM configuration with LangSmith tracing
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment variables

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and add your API keys:

```bash
copy .env.example .env
```

5. Run the app:

```bash
streamlit run app.py
```

## Environment Variables

Set the following in `.env`:

```text
GROQ_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=Hospital-RAG
LANGCHAIN_TRACING_V2=true
```

## Usage

1. Upload a hospital information PDF in the sidebar.
2. Enter a question in the main page input.
3. Click `Ask` to retrieve and generate an answer.

## Notes

- The app only answers using context from the uploaded PDF.
- If the answer cannot be found, it returns:
  `Sorry, I couldn't find that information in the hospital document.`
- Embeddings are persisted in `chroma_db/`.
