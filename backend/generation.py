# generation.py

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.5-flash"

genai_client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_answer(query: str, contexts: list):
    context_text = ""

    for ctx in contexts:
        context_text += f"\n[Page {ctx['page']}]\n{ctx['text']}\n"

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
Mention page numbers in your answer like (Page X).

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = genai_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text