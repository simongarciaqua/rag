
# Debugging Google Generative AI dimensions
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

text = "Hello world"
model = "models/text-embedding-004"

print(f"Embedding with model: {model}")

# Test 1: Simple embedding
try:
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document",
        output_dimensionality=1024
    )
    vec = result['embedding']
    print(f"Single text dimension: {len(vec)}")
except Exception as e:
    print(f"Error single: {e}")

# Test 2: Batch embedding
try:
    batch = ["Hello world", "Another text"]
    result = genai.embed_content(
        model=model,
        content=batch,
        task_type="retrieval_document",
        output_dimensionality=1024
    )
    vecs = result['embedding']
    print(f"Batch text dimension: {len(vecs[0])}")
except Exception as e:
    print(f"Error batch: {e}")
