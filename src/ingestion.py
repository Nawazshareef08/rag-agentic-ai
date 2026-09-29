import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def run_ingestion(pdf_path: str, index_name: str):
    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings and storing them in Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "Ebook-Agentic-AI.pdf",
    )

    index_name = os.getenv(
        "PINECONE_INDEX_NAME",
        "agentic-ai-local",
    )

    run_ingestion(pdf_path, index_name)
