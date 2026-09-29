from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask_question

app = FastAPI(
    title="Agentic AI RAG API",
    description="RAG API for the Agentic AI eBook",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    result = ask_question(request.question)

    return {
        "query": request.question,
        "final_answer": result["final_answer"],
        "retrieved_context_chunks": result["retrieved_context_chunks"],
        "confidence_score": result["confidence_score"],
    }
