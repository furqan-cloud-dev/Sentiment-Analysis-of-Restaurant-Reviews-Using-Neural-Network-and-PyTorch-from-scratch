# Step 3: sentiment_api.py – FastAPI Server

# sentiment_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import nltk

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Reuse same dictionaries and model
word_dict = {
    "strongly_positive": {"amazing", "fantastic", "outstanding", "superb", "excellent"},
    "positive": {"good", "great", "love", "like", "nice"},
    "strongly_negative": {"horrible", "terrible", "worst", "disgusting", "vile"},
    "negative": {"bad", "poor", "hate", "dislike", "not good"},
    "neutral": {"okay", "average", "fine", "normal"},
    "abusive": {"stupid", "idiot", "dumb", "shut up", "nonsense"},
    "constructive": {"should", "could", "suggest", "recommend", "consider", "perhaps", "improve"}
}

label_list = ["negative", "positive", "neutral", "constructive feedback"]

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def extract_features(tokens):
    return [sum(w in word_dict[cat] for w in tokens) for cat in word_dict]

class SentimentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

model = SentimentNet()
model.load_state_dict(torch.load("sentiment_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI()

class RequestBody(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(req: RequestBody):
    tokens = preprocess(req.text)
    features = torch.tensor(extract_features(tokens), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(features)
        pred = torch.argmax(out, dim=1).item()
    return {"sentiment": label_list[pred]}
