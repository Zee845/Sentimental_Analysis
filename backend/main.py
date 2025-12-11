import uuid
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 1. Initialize the AI Model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Initialize ChromaDB Client
# We connect to the separate container named "chromadb"
try:
    chroma_client = chromadb.HttpClient(host='chromadb', port=8000)
    collection = chroma_client.get_or_create_collection(name="sentiment_history")
    print("Connected to ChromaDB successfully!")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB. Is the container running? {e}")
    collection = None

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    # Run the AI model
    result = sentiment_pipeline(request.text)[0]
    label = result['label']
    score = round(result['score'], 4)
    
    # Store result in Vector Database (ChromaDB)
    if collection:
        try:
            collection.add(
                documents=[request.text],
                metadatas=[{"label": label, "score": score}],
                ids=[str(uuid.uuid4())]  # Generate a unique ID
            )
            print(f"Saved to DB: {request.text}")
        except Exception as e:
            print(f"Failed to save to DB: {e}")

    return {
        "label": label,
        "score": score,
        "db_status": "Saved to Vector DB" if collection else "DB Unavailable"
    }