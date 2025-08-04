"""
Train, Test and Predict Model
Update Preprocessing
"""

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
