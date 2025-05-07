import os
import re
import string
import joblib # type: ignore
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# Add custom nltk_data path if you downloaded manually
nltk.data.path.append(r"utils\nltk")

class PhishingTextScanner:
    def __init__(self):
        # Load pre-trained model and vectorizer
        model_path = os.path.join("models", "phishing_text_model.pkl")
        vectorizer_path = os.path.join("models", "vectorizer.pkl")

        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Load stop words
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        # Lowercase
        text = text.lower()

        # Remove URLs, mentions, hashtags, and punctuation
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        return " ".join(filtered_tokens)

    def check_text(self, text):
        cleaned_text = self.clean_text(text)
        vectorized_input = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_input)[0]
        label = "phishing" if prediction == 1 else "safe"
        confidence = self.model.predict_proba(vectorized_input)[0][prediction]

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "result": label,
            "confidence": round(confidence, 2)
        }
