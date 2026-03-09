from llama_cpp import Llama

llm = Llama(
    model_path=r"C:\Users\Laptopster\docsclassify\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=1024,
    verbose=False
)

PROMPT = """
Convert the user query into a concise semantic search query.

Keep:
- names
- numbers
- entities
- document intent

Do not explain.
Return only the rewritten query.

User query: {query}
Search query:
"""

def encode_query(query):
    response = llm(
        PROMPT.format(query=query),
        max_tokens=40,
        temperature=0
    )
    return response["choices"][0]["text"].strip()
