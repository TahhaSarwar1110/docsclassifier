from llama_cpp import Llama
import json

llm = Llama(
    model_path=r"C:\Users\Laptopster\docsclassify\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    verbose=False
)

PROMPT = """
You are a document search interpreter.

Convert the user query into JSON.

Fields:
- type: resume | invoice | general
- experience_years: number or null
- amount: number or null

Return ONLY JSON.

Query: {query}
JSON:
"""

def parse_query(query):
    response = llm(
        PROMPT.format(query=query),
        max_tokens=120,
        temperature=0
    )

    text = response["choices"][0]["text"].strip()

    try:
        return json.loads(text)
    except:
        return {"type": "general", "experience_years": None, "amount": None}
