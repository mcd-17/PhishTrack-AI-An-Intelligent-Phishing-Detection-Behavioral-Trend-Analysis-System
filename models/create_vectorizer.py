# models/create_vectorizer.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Example dataset to create the vectorizer
newsgroups = fetch_20newsgroups(subset='train')
texts = newsgroups.data

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on the training data (i.e., text corpus)
X_train_tfidf = vectorizer.fit_transform(texts)

# Save the vectorizer to a pickle file
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Vectorizer saved as models/vectorizer.pkl")
