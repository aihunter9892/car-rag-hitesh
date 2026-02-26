# main.py

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.retrieval import retrieve
from backend.generation import generate_answer

app = FastAPI()

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.post("/ask")
def ask_question(data: QueryRequest):
    contexts = retrieve(data.query)
    answer = generate_answer(data.query, contexts)

    return {
        "answer": answer,
        "sources": [
            {
                "page": ctx["page"],
                "score": ctx["score"]
            }
            for ctx in contexts
        ]
    }