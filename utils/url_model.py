import joblib
import os
from urllib.parse import urlparse

class PhishingURLDetector:
    def __init__(self):
        # Load trained phishing detection model and vectorizer
        model_path = os.path.join("models", "phishing_url_model.pkl")
        self.model = joblib.load(model_path)

    def extract_features(self, url):
        parsed_url = urlparse(url)
        domain_length = len(parsed_url.netloc)
        url_length = len(url)
        has_https = int(url.startswith("https"))
        num_dots = url.count(".")
        suspicious_words = any(word in url.lower() for word in ["login", "verify", "secure", "update", "account"])
        
        # Return feature list
        return [[domain_length, url_length, has_https, num_dots, suspicious_words]]

    def check_url(self, url):
        features = self.extract_features(url)
        prediction = self.model.predict(features)[0]
        label = "phishing" if prediction == 1 else "safe"
        confidence = self.model.predict_proba(features)[0][prediction]

        return {
            "url": url,
            "result": label,
            "confidence": round(confidence, 2)
        }
