import gradio as gr
import requests
import pandas as pd

# Define URLs
BACKEND_ANALYZE_URL = "http://backend:8000/analyze"
BACKEND_HISTORY_URL = "http://backend:8000/history"

def get_sentiment(text):
    if not text:
        return "Please enter text."
    try:
        payload = {"text": text}
        response = requests.post(BACKEND_ANALYZE_URL, json=payload)
        data = response.json()
        return f"Sentiment: {data['label']}\nConfidence: {data['score']}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_history():
    try:
        response = requests.get(BACKEND_HISTORY_URL)
        data = response.json()
        
        # Check if data is a list (valid) or dict (error)
        if isinstance(data, dict) and "error" in data:
            return pd.DataFrame({"Error": [data["error"]]})
            
        if not data:
            return pd.DataFrame({"Status": ["No data found in DB"]})

        # Convert to Pandas Dataframe for a nice table
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# Create the Gradio App with Tabs
with gr.Blocks(title="Sentiment AI & Vector DB") as app:
    gr.Markdown("# 🧠 AI Sentiment Analyzer + Vector Database")
    
    with gr.Tabs():
        # Tab 1: The Analyzer
        with gr.TabItem("Analyze Text"):
            with gr.Row():
                input_text = gr.Textbox(lines=2, placeholder="Type something here...", label="Input Text")
                output_text = gr.Textbox(label="Result")
            submit_btn = gr.Button("Analyze & Save")
            submit_btn.click(get_sentiment, inputs=input_text, outputs=output_text)

        # Tab 2: The Database Viewer
        with gr.TabItem("Database History"):
            gr.Markdown("Click refresh to see what is stored inside ChromaDB (Vector Store).")
            refresh_btn = gr.Button("Refresh Data")
            history_table = gr.Dataframe(label="Stored Vectors")
            refresh_btn.click(get_history, outputs=history_table)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)