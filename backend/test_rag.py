from retrieval import retrieve
from generation import generate_answer

query = "What is this document about?"

contexts = retrieve(query)
answer = generate_answer(query, contexts)

print("\nAnswer:\n")
print(answer)