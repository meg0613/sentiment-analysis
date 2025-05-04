import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle, os

# Load the dataset
data = pd.read_csv("IMDB Dataset.csv")

# Check the actual column names (optional, remove later)
print(data.columns)

# Use the correct column names from your CSV
X = data['review']
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")

