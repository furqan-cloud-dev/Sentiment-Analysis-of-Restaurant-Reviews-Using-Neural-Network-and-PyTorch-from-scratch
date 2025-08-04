# Efficiently Handle Big Datasets

# Heuristic Labeling Function (if unlabeled)
#If you don’t have labeled data:

def infer_label(text):
    # simple heuristic or rules
    if any(w in text.lower() for w in ["worst", "awful", "disgusting", "bad"]):
        return "negative"
    elif any(w in text.lower() for w in ["excellent", "perfect", "amazing", "great"]):
        return "positive"
    elif any(w in text.lower() for w in ["could", "should", "suggest"]):
        return "constructive feedback"
    else:
        return "neutral"


# Use chunks when reading huge CSV/JSON files:

chunks = pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=50000)

texts = []
labels = []

for chunk in chunks:
    for _, row in chunk.iterrows():
        if 'food' in row['text'].lower() or 'restaurant' in row['text'].lower():
            texts.append(row['text'])
            # Add your labeling logic based on heuristics
            labels.append(infer_label(row['text']))



# Save Model to Google Drive (optional)

from google.colab import drive
drive.mount('/content/drive')

torch.save(model.state_dict(), "/content/drive/MyDrive/sentiment_model.pth")

# Bonus: Vectorize Feature Extraction (Speed Boost)
# Instead of counting words one by one, you can vectorize it:

import numpy as np
from data import word_dict

def extract_features_fast(tokens):
    features = np.zeros(len(word_dict))
    word_set = set(tokens)
    for i, category in enumerate(word_dict):
        features[i] = len(word_set & word_dict[category])
    return features.tolist()



