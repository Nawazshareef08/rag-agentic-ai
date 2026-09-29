import os
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()


class GraphState(TypedDict):
    question: str
    context: str
    answer: str


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_name = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-local",
)

vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


def retrieve(state: GraphState):
    question = state["question"]

    documents = vector_store.similarity_search(
        question,
        k=4,
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "context": context,
    }


def generate_answer(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions
about the Agentic AI eBook.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I couldn't find that information in the provided eBook."

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": state["context"],
            "question": state["question"],
        }
    )

    return {
        "answer": response.content,
    }


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

rag_graph = workflow.compile()


def ask_question(question: str) -> str:
    result = rag_graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
        }
    )

    return result["answer"]
