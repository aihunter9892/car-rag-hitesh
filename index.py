from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()  # THIS IS REQUIRED


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


index_name = "rag-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,   # Gemini embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Index ready.")