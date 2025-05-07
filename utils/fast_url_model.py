# utils/fast_url_model.py

import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def extract_url_features(url):
    """Extracts lightweight and fast URL features for classification."""
    return {
        'url_length': len(url),
        'has_https': int('https' in url),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_at': url.count('@'),
        'has_ip': int(bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url))),
        'has_suspicious_words': int(any(w in url.lower() for w in ['login', 'verify', 'bank', 'update'])),
    }


def prepare_dataset(df, url_column='url', label_column='label'):
    """Converts raw URL dataset into feature DataFrame."""
    features = df[url_column].apply(lambda x: pd.Series(extract_url_features(x)))
    features[label_column] = df[label_column]
    return features


def train_fast_url_model(csv_path, model_path):
    """Trains and saves a fast RandomForest phishing model."""
    df = pd.read_csv(csv_path)
    dataset = prepare_dataset(df)

    X = dataset.drop(columns=['label'])
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model Evaluation:\n", classification_report(y_test, preds))

    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")


def predict_url(url, model_path):
    """Loads model and predicts phishing status of a given URL."""
    model = joblib.load(model_path)
    features = pd.DataFrame([extract_url_features(url)])
    prediction = model.predict(features)
    return prediction[0]
