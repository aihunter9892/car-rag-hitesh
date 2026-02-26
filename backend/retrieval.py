# retrieval.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "rag-index"
NAMESPACE = "ai_non_tech"
EMBED_MODEL = "gemini-embedding-001"

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

genai_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_query(query: str):
    result = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )
    return result.embeddings[0].values


def retrieve(query: str, top_k: int = 3):
    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE
    )

    contexts = []
    for match in results["matches"]:
        contexts.append({
            "text": match["metadata"]["text"],
            "page": match["metadata"]["page"],
            "score": match["score"]
        })

    return contexts