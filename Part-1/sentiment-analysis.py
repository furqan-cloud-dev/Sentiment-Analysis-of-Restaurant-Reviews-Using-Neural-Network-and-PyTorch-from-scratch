

# 1. Define Updated Word Dictionaries

word_dict = {
    "strongly_positive": {"amazing", "fantastic", "outstanding", "superb", "excellent"},
    "positive": {"good", "great", "love", "like", "nice"},
    "strongly_negative": {"horrible", "terrible", "worst", "disgusting", "vile"},
    "negative": {"bad", "poor", "hate", "dislike", "not good"},
    "neutral": {"okay", "average", "fine", "normal"},
    "abusive": {"stupid", "idiot", "dumb", "shut up", "nonsense"},
    "constructive": {"should", "could", "suggest", "recommend", "consider", "perhaps"}
}


# 2. Preprocessing Function

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return filtered


# 3. Feature Extraction (7 features)

def extract_features(tokens):
    features = []
    for category in ["negative", "strongly_negative", "abusive", "positive", "strongly_positive", "neutral", "constructive"]:
        features.append(sum(word in word_dict[category] for word in tokens))
    return features  # 7-element vector



# Labels and Dataset Class


from torch.utils.data import Dataset

# New output classes: 0=negative, 1=positive, 2=neutral, 3=constructive
label_map = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
    "constructive feedback": 3
}

class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = []
        for text, label in zip(texts, labels):
            tokens = preprocess(text)
            features = extract_features(tokens)
            self.data.append((torch.tensor(features, dtype=torch.float32), label_map[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 5. Neural Network

import torch.nn as nn
import torch.nn.functional as F

class SentimentNet(nn.Module):
    def __init__(self):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(7, 16)     # Input: 7 features
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)     # Output: 4 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


# 6. Training Loop (with 4-class NLLLoss)
from torch.utils.data import DataLoader
import torch.optim as optim


def train_model(dataset):
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SentimentNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    for epoch in range(10):
        total_loss = 0
        for features, label in loader:
            output = model(features)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

    return model


# 7. Prediction Function

def predict(model, text):
    tokens = preprocess(text)
    features = torch.tensor(extract_features(tokens), dtype=torch.float32).unsqueeze(0)
    output = model(features)
    pred = torch.argmax(output, dim=1).item()
    return ["negative", "positive", "neutral", "constructive feedback"][pred]









