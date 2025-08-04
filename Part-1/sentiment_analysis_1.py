"""
Unlock The Mystery Behind Artificial Neural Network
NLP Sentiment Analysis Of Food & Restaurants Reviews

🧠 NLP + Python + PyTorch = Neural Network
Practical coding with Python
DEEP LEARNING
Tensors to Training Neural Network on GPU

Train a Neural Network in PyTorch: A Complete Beginner's Walkthrough

Step-by-Step guide with explanations Of Python Code Examples
👇 Get your instant download now for this AI Magazine (PDF Format):
🔗 https://aicampusmagazines.gumroad.com/l/loukc

✨ Brought to you by AI Campus – Your gateway to AI knowledge.

Contact Development Team:
Muhammad Furqan
Software Developer | AI/ML Solution Architect
Email: furqan.cloud.dev@gmail.com
LinkedIn: https://www.linkedin.com/in/muhammad-furqan-0436a6355/
"""


from data import word_dict

# Preprocessing Function
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


# Feature Extraction (7 features)
def extract_features(tokens):
    features = []
    tokens_set = set(tokens)
    for words_list in word_dict.values():
        # intersection for common elements - matched words
        matched_words_count = len(tokens_set.intersection(words_list))
        features.append(matched_words_count)

    return features  # 7-element vector


 review_text = ("""The menu offers a wide variety of options, and everything we tried was delicious.
               I especially loved the creative twists on classic dishes""")

tokens = preprocess(review_text)
features = extract_features(tokens)

print(review_text)
print("Tokens: ", tokens)
print("Tokens Count:",len(tokens))
print(features)

categories = list(word_dict.keys())
print(categories)

###########################################

# Plotting Graph
import matplotlib.pyplot as plt

# X-axis values (emotional scale)
x_values = [-3, -2, -1, 0, 1, 2, 3]
# Category labels mapped to x-axis positions
categories_labels = list(word_dict.keys())

# Word counts in each category
input_features = features

# Plot the data
plt.figure(figsize=(10, 5))
plt.bar(x_values, input_features, color='skyblue', edgecolor='black')

# Set category names as tick labels
plt.xticks(ticks=x_values, labels=categories_labels, rotation=30)

# Add labels and title
plt.xlabel("Sentiment Category")
plt.ylabel("Word Count")
plt.title("Word Count per Review Sentiment Category")

# Show value on top of each bar
for i, count in enumerate(input_features):
    plt.text(x_values[i], count + 0.2, str(count), ha='center')

# Display the plot
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.show()
plt.savefig("plot.png")
