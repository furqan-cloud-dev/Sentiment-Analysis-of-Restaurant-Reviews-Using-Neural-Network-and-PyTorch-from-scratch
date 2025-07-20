# Scale this sentiment analysis pipeline on Google Colab Free GPU, even with a large dataset of food & restaurant reviews. 
## Here's exactly how to adapt everything:


🍔 1. ✅ Get a Large Dataset of Restaurant Reviews
Option A: Use Yelp Open Dataset

!wget https://s3.amazonaws.com/yelp-dataset/yelp_academic_dataset_review.json


This file contains ~8M reviews. You can filter only those with "categories" related to food/restaurants.

Option B: Use Kaggle Google Review Datasets
You can use:

Restaurant Reviews - Kaggle
https://www.kaggle.com/datasets/saikrishnab/google-reviews-restaurant


To access Kaggle datasets:

!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d saikrishnab/google-reviews-restaurant
!unzip google-reviews-restaurant.zip


🧠 2. Train on Colab GPU
✅ GPU Setup
In Colab:

Runtime > Change runtime type > GPU


In your train_model.py, add:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


Update your training loop:
for features, label in loader:
    features, label = features.to(device), label.to(device)
    output = model(features)
    ...



🧪 3. Efficiently Handle Big Datasets

Use chunks when reading huge CSV/JSON files:
chunks = pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=50000)

texts = []
labels = []

for chunk in chunks:
    for _, row in chunk.iterrows():
        if 'food' in row['text'].lower() or 'restaurant' in row['text'].lower():
            texts.append(row['text'])
            # Add your labeling logic based on heuristics
            labels.append(infer_label(row['text']))



🌐 6. FastAPI Deployment in Colab (GPU not needed here)

!pip install fastapi uvicorn pyngrok nest-asyncio


In Colab:

import nest_asyncio
from pyngrok import ngrok
import uvicorn

nest_asyncio.apply()
public_url = ngrok.connect(8000)
print(public_url)

uvicorn.run(app="sentiment_api:app", host="0.0.0.0", port=8000)



✅ What You Now Have


| Component          | Status                   |
| ------------------ | ------------------------ |
| Colab GPU training | ✅ Yes                    |
| Large review data  | ✅ Yelp / Kaggle          |
| Feature extraction | ✅ Word-count vector      |
| Scalable labeling  | ✅ Rule-based or ML       |
| Model saving       | ✅ To disk or Drive       |
| FastAPI inference  | ✅ In Colab via `pyngrok` |




🧠 What's Included

| Section                 | Description                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------- |
| ✅ Dataset Loader        | Downloads & reads sample restaurant review dataset (Kaggle-style CSV or JSON chunk) |
| 🔤 Preprocessing        | Text cleaning, stopword removal                                                     |
| 📊 Feature Extraction   | Counts of words from 7 sentiment categories                                         |
| 🧠 PyTorch Model        | 3-layer neural network with 7 → 4 classification                                    |
| 🚀 Training (GPU-ready) | Run on Colab GPU with Adam optimizer                                                |
| 💾 Model Saving         | Save `.pth` to Colab or Google Drive                                                |
| 🌐 FastAPI Server       | Deployed using `FastAPI + pyngrok`                                                  |
| ⚡ Test Endpoint         | Send prediction request to live public URL                                          |




