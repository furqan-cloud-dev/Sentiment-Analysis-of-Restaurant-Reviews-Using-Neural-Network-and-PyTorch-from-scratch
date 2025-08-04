"""
Train & Save Model
"""

# train_model.py

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Output Classification
label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "constructive feedback": 3
}


class ReviewDataset(Dataset):
    def __init__(self, df):
        self.data = []
        for _, row in df.iterrows():
            tokens = preprocess(row['text'])
            features = extract_features(tokens)
            label = label_mapping[row['label']]
            self.data.append((torch.tensor(features, dtype=torch.float32), label))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def train():
    df = pd.read_csv("reviews.csv")
    dataset = ReviewDataset(df)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SentimentClassifierNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(100):
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
