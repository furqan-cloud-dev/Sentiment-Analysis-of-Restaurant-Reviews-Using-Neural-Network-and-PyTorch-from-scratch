# train_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Word categories ---
word_dict = {
    "strongly_positive": {"amazing", "fantastic", "outstanding", "superb", "excellent"},
    "positive": {"good", "great", "love", "like", "nice"},
    "strongly_negative": {"horrible", "terrible", "worst", "disgusting", "vile"},
    "negative": {"bad", "poor", "hate", "dislike", "not good"},
    "neutral": {"okay", "average", "fine", "normal"},
    "abusive": {"stupid", "idiot", "dumb", "shut up", "nonsense"},
    "constructive": {"should", "could", "suggest", "recommend", "consider", "perhaps", "improve"}
}

label_map = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
    "constructive feedback": 3
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def extract_features(tokens):
    return [sum(w in word_dict[cat] for w in tokens) for cat in word_dict]

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.data = []
        for _, row in df.iterrows():
            tokens = preprocess(row['text'])
            features = extract_features(tokens)
            label = label_map[row['label']]
            self.data.append((torch.tensor(features, dtype=torch.float32), label))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

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

def train():
    df = pd.read_csv("reviews.csv")
    dataset = ReviewDataset(df)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SentimentNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(20):
        total_loss = 0
        for features, label in loader:
            output = model(features)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "sentiment_model.pth")
    print("✅ Model saved to sentiment_model.pth")

if __name__ == "__main__":
    train()
