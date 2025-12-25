import uuid
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 1. Initialize AI Model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Initialize ChromaDB
try:
    chroma_client = chromadb.HttpClient(host='chromadb', port=8000)
    collection = chroma_client.get_or_create_collection(name="sentiment_history")
    print("Connected to ChromaDB successfully!")
except Exception as e:
    print(f"Warning: DB Connection failed. {e}")
    collection = None

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Backend is running"}

# --- NEW: Endpoint to fetch history from DB ---
@app.get("/history")
def get_history():
    if not collection:
        return {"error": "Database not connected"}
    
    # Fetch the last 10 documents
    try:
        data = collection.peek(limit=10)
        # Simplify the data for the frontend
        history = []
        if data['ids']:
            for i in range(len(data['ids'])):
                history.append({
                    "Text": data['documents'][i],
                    "Sentiment": data['metadatas'][i]['label'],
                    "Score": data['metadatas'][i]['score']
                })
        return history
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    label = result['label']
    score = round(result['score'], 4)
    
    # Save to ChromaDB
    if collection:
        try:
            collection.add(
                documents=[request.text],
                metadatas=[{"label": label, "score": score}],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"DB Error: {e}")

    return {"label": label, "score": score}