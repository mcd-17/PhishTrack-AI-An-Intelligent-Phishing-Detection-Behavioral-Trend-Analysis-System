# text_model_train.py
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Login now to claim your prize!",
    "Reset your password immediately",
    "Meeting at 3pm in office",
    "Your account has been suspended",
    "Hey, want to grab coffee later?"
]

labels = [1, 1, 0, 1, 0]  # 1 = phishing, 0 = safe

model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

joblib.dump(model, 'models/phishing_text_model.pkl')
