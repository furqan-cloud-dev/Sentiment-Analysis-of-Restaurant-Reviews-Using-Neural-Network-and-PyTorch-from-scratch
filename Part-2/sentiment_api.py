"""
RESTful API Access For The Model
FastAPI - Server
"""

# sentiment_api.py

"""
To run a FastAPI server, follow these steps: 
    Install FastAPI and Uvicorn:
    pip install "fastapi[standard]"

This command installs FastAPI along with Uvicorn, which is a high-performance ASGI server commonly used with FastAPI
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch


label_list = ["negative", "positive", "neutral", "constructive feedback"]

# Neural Network - Sentiment Classifier
model = SentimentClassifierNN()
model.load_state_dict(torch.load("sentiment_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI()

class RequestBody(BaseModel):
    text: str


@app.get("/")
async def home():
    return {"message": "SentimentClassifier Neural Network Model"}


@app.post("/predict")
async def predict_sentiment(req: RequestBody):
    tokens = preprocess(req.text)
    features = torch.tensor(extract_features(tokens), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(features)
        pred = torch.argmax(out, dim=1).item()
    return {"sentiment": label_list[pred]}


"""
Run the FastAPI server.
Open your terminal or command prompt in the directory where your sentiment_api.py file is located and execute the following command:

    uvicorn sentiment_api:app --reload
"""