# utils/intent_classifier.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import logging

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Logging configuration
logging.basicConfig(level=logging.INFO)

class IntentClassifier:
    def __init__(self, model_path=None):
        """
        Initializes the classifier, and optionally loads a pre-trained model from the specified path.
        """
        self.model = None
        if model_path:
            self.load_model(model_path)
        else:
            # If no model path is provided, a default model will be trained
            self.model = self.train_default_model()

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def train_default_model(self):
        """
        Train a default intent classification model using logistic regression.
        """
        # Example training data (You should replace this with real training data)
        texts = [
            "You must act now! Limited time offer!",
            "Urgent! Your account has been compromised, click here to fix it!",
            "This is a friendly reminder to complete your form.",
            "We are happy to inform you that your prize has been won!"
        ]
        labels = ["phishing", "phishing", "safe", "safe"]  # Labels: phishing vs safe

        # Vectorizer and classifier pipeline
        model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )

        # Train the model
        model.fit(texts, labels)
        logging.info("Model trained on default data.")

        return model

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified file path.
        """
        self.model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}.")

    def classify_intent(self, text):
        """
        Classifies the intent of the given text as either phishing or safe.
        """
        if self.model:
            prediction = self.model.predict([text])[0]
            logging.info(f"Predicted intent for text: {prediction}")
            return prediction
        else:
            logging.error("Model is not loaded or trained.")
            return None

    def check_urgency(self, text):
        """
        Uses sentiment analysis to detect urgency or emotional manipulation.
        A higher negative sentiment could indicate phishing-like urgency.
        """
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        if sentiment['compound'] < -0.5:
            logging.info("Urgency detected in text.")
            return "high_urgency"
        else:
            logging.info("No urgency detected.")
            return "low_urgency"

    def analyze_intent(self, text):
        """
        Analyzes the intent and urgency of the given text.
        Combines classification and sentiment analysis.
        """
        intent = self
