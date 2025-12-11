import gradio as gr
import requests

# The Docker service name for the backend is "backend"
BACKEND_URL = "http://backend:8000/analyze"

def get_sentiment(text):
    if not text:
        return "Please enter text."
    
    try:
        # Send request to Backend container
        payload = {"text": text}
        response = requests.post(BACKEND_URL, json=payload)
        data = response.json()
        
        # Format output
        return f"Sentiment: {data['label']}\nConfidence Score: {data['score']}"
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

# Create Gradio Interface
iface = gr.Interface(
    fn=get_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Deep Learning Sentiment Analyzer",
    description="Enter text to analyze sentiment using a DistilBERT Transformer model running in Docker."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)