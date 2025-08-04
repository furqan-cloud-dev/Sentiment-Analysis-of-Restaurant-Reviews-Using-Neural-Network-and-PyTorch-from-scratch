"""
Train, Test and Predict Model
Training DataSet: "reviews.csv"
- - - - - - - - - - - - -
Updated Preprocessing
Integrated Vector Normalization
"""

from data import word_dict

# Preprocessing Function
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Updated - Text Processing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()

    # Build n-grams up to trigrams
    # To match multi-word phrases like "amazing service", "not good", "should add more" etc.
    ngrams = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        ngrams.append(bigram)
    for i in range(len(tokens) - 2):
        trigram = tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2]
        ngrams.append(trigram)

    # Remove stop words from single words list
    filtered = [word for word in tokens if word not in stop_words]
    combined_tokens = filtered + ngrams
    return combined_tokens

# Feature Extraction (7 features)
def extract_features(tokens):
    features = []
    tokens_set = set(tokens)
    for words_list in word_dict.values():
        # intersection for common elements - matched words
        matched_words_count = len(tokens_set.intersection(words_list))
        features.append(matched_words_count)

    return features  # 7-element vector



# Neural Network - Sentiment Classifier
import torch.nn as nn
import torch.nn.functional as F

# Inheritance/Sub-Class from PyTorch's: torch.nn.Module (Base Class Of Neural Network)
class SentimentClassifierNN(nn.Module):
    def __init__(self):
        super(SentimentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(7, 16)     # Input: 7 features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)     # Output: 4 classes

    # Must implement a forward(input) method
    # Transforming the input tensor(s) into output tensor(s)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)  # Softmax Activation for Probability distribution




# Output Classification
label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "constructive feedback": 3
}

import numpy as np

def extract_reviews_features(texts):
    """Extract various text-based features"""
    reviews_features = []

    for text in texts:
        tokens = preprocess(text)
        features = extract_features(tokens)
        reviews_features.append(features)

    return np.array(reviews_features)


# Min-Max Normalization - GLOBALS Reuse in Normalization Function
min_vals = None
max_vals = None

def initialize_min_max(features):
    global min_vals, max_vals
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)

def min_max_normalize(X):
    global min_vals, max_vals
    if min_vals is not None and max_vals is not None:
        return (X - min_vals) / (max_vals - min_vals + 1e-8) # eps=1e-8 - avoid divison by zero
    raise ValueError("min_vals and max_vals are not initialized")


import pandas as pd

def process_csv_data(csv_file_path):
    """Process CSV data and create feature tensors"""
    df = pd.read_csv(csv_file_path) # Load CSV data

    # Extract texts and labels
    texts = df['text'].tolist()
    original_labels = df['label'].tolist()

    # Map labels to 4 classes
    mapped_labels = [label_mapping.get(label, 1) for label in original_labels]  # default to neutral

    # Extract text features (7 features)
    features = extract_reviews_features(texts)

    # During training
    initialize_min_max(features)

    # normalization
    normalized_features = min_max_normalize(features)

    # Convert to tensors
    X_tensor = torch.FloatTensor(normalized_features)
    y_tensor = torch.LongTensor(mapped_labels)

    return X_tensor, y_tensor, df


from sklearn.model_selection import train_test_split

# Training function
def train_model(X_tensor, y_tensor, epochs=100, lr=0.01):
    """Train the neural network"""
    model = SentimentClassifierNN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    model.eval() # Test the model
    with torch.no_grad():
        test_output = model(X_test)
        test_pred = torch.argmax(test_output, dim=1)
        accuracy = (test_pred == y_test).float().mean()
        print("--------------------------------")
        print(f'Test Accuracy: {accuracy:.4f}')

    return model


def predict_new_text(model, text):
    """Predict sentiment for new text"""
    # Extract text features (7 features)
    tokens = preprocess(text)
    features = extract_features(tokens)
    print(features)

    features_array = np.array(features)
    # Reshape for single sample (1 sample, 7 features)
    features_2d = features_array.reshape(1, -1)

    # normalization
    normalized_features = min_max_normalize(features_2d)

    # Convert to tensor
    X_tensor = torch.FloatTensor(normalized_features)
    # print(tokens)
    print("Tokens Count: ", len(tokens))
    print(X_tensor)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Map back to sentiment labels
    sentiment_map = {0: ' negative', 1: 'neutral', 2: 'positive', 3: 'constructive feedback'}
    return sentiment_map[prediction]


# Example usage
if __name__ == "__main__":
    # Process your CSV data
    print("Processing CSV data...")
    X_tensor, y_tensor, df = process_csv_data('reviews.csv')

    print(f"Features shape: {X_tensor.shape}")
    print(f"Labels shape: {y_tensor.shape}")
    print(f"Unique labels in dataset: {torch.unique(y_tensor)}")

    # Train the model
    print("\nTraining neural network...")
    trained_model = train_model(X_tensor, y_tensor)

    # Test with new examples
    print("\nTesting predictions:")
    print("----------------------")
    test_texts = [
        "This restaurant is amazing",
        "The service was terrible",
        "It was okay, nothing special",
        "You should add more vegetarian options",
        """The menu offers a wide variety of options, and everything we tried was delicious.
        I especially loved the creative twists on classic dishes"""
    ]

    for text in test_texts:
        prediction = predict_new_text(trained_model, text)
        print(f"Text: '{text}' -> Predicted: {prediction}")
        print("------------------------")

    print(
        "Label mapping: negative -> 0, neutral -> 1, positive -> 2", "constructive feedback -> 3")

