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
    answer: str


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    answer = ask_question(request.question)

    return {
        "answer": answer
    }
