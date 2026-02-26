from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")

index.delete(delete_all=True, namespace="ai_non_tech")

print("Namespace cleared.")