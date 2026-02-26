# ingest_pdf.py

import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from pypdf import PdfReader

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
INDEX_NAME = "rag-index"
NAMESPACE = "ai_non_tech"
PDF_PATH = r"F:\inclass\RAG\AI for non tech.pdf"
EMBED_MODEL = "gemini-embedding-001"

# ---------------------------------------------------
# Initialize Clients
# ---------------------------------------------------
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Initializing Gemini...")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------------------------------------------
# Extract PDF by Page (Best for RAG)
# ---------------------------------------------------
def extract_pdf_chunks(path):
    reader = PdfReader(path)
    chunks = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text and len(page_text.strip()) > 50:
            chunks.append({
                "text": page_text.strip(),
                "page": page_num + 1
            })

    return chunks


# ---------------------------------------------------
# Embed Text using Gemini (3072 dimensions)
# ---------------------------------------------------
def embed_text(text):
    result = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return result.embeddings[0].values


# ---------------------------------------------------
# Main Ingestion Function
# ---------------------------------------------------
def ingest():
    print("\nExtracting PDF...")
    page_chunks = extract_pdf_chunks(PDF_PATH)

    if not page_chunks:
        print("No text extracted from PDF.")
        return

    print(f"Total Pages/Chunks: {len(page_chunks)}")

    vectors = []

    for chunk in page_chunks:
        embedding = embed_text(chunk["text"])

        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": os.path.basename(PDF_PATH)
            }
        })

        print(f"Embedded page {chunk['page']}")

    print("\nUpserting to Pinecone...")
    index.upsert(vectors=vectors, namespace=NAMESPACE)

    print("\n✅ Ingestion Complete.")
    print(f"Namespace: {NAMESPACE}")
    print(f"Total vectors inserted: {len(vectors)}")


# ---------------------------------------------------
# Run Script
# ---------------------------------------------------
if __name__ == "__main__":
    ingest()