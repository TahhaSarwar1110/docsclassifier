from llama_cpp import Llama
import os

model_path = os.path.join(
    os.getcwd(),
    "models",
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

print("Loading model from:", model_path)

llm = Llama(
    model_path=model_path,
    verbose=False
)

print("Model loaded successfully!")
