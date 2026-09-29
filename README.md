# Agentic AI RAG

A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from the Agentic AI eBook.

## Features

- PDF document ingestion
- Text chunking
- Local Hugging Face embeddings
- Pinecone vector database
- LangGraph workflow
- Local Llama 3.2 LLM through Ollama
- FastAPI REST API
- Swagger API documentation

## Project Structure

rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── ingestion.py
│   ├── graph.py
│   └── api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

## Requirements

- Python 3.12
- Pinecone account
- Ollama

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-agentic-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install Ollama

Install Ollama and download the model:

```powershell
ollama pull llama3.2:3b
```

### 6. Configure environment variables

Create a `.env` file:

```text
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-local
```

Never commit `.env` to GitHub.

### 7. Ingest the PDF

Place `Ebook-Agentic-AI.pdf` inside the `data` folder and run:

```powershell
python -m src.ingestion
```

### 8. Start the API

```powershell
uvicorn src.api:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Swagger documentation:

`http://127.0.0.1:8000/docs`

## Example Request

POST `/chat`

```json
{
  "question": "What is agentic AI?"
}
```

Example response:

```json
{
  "answer": "According to the provided context, Agentic AI is a type of artificial intelligence that enables autonomous decision-making and action."
}
```

## Architecture

PDF → Text Chunks → Local Embeddings → Pinecone → Retrieval → LangGraph → Llama 3.2 → FastAPI

## License

This project is for educational purposes.
