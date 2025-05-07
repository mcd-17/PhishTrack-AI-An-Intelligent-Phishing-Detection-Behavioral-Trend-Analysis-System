# utils/ensemble_model.py

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import numpy as np

class EnsembleModel:
    def __init__(self):
        # Initialize the base models
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
        self.svc_model = SVC(probability=True, random_state=42)

        # Initialize the ensemble classifier with Voting Classifier
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('decision_tree', self.dt_model),
                ('random_forest', self.rf_model),
                ('logistic_regression', self.lr_model),
                ('svc', self.svc_model)
            ],
            voting='soft'  # 'soft' uses predicted probabilities, 'hard' uses predicted labels
        )

    def train(self, X_train, y_train):
        """
        Train the ensemble model using the provided training data.
        """
        print("Training the ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        joblib.dump(self.ensemble_model, 'models/ensemble_model.pkl')  # Save the trained model
        print("Ensemble model trained and saved as 'ensemble_model.pkl'")

    def predict(self, X):
        """
        Predict using the trained ensemble model.
        """
        return self.ensemble_model.predict(X)

    def predict_proba(self, X):
        """
        Get the probability predictions using the trained ensemble model.
        """
        return self.ensemble_model.predict_proba(X)

    def load_model(self, model_path='models/ensemble_model.pkl'):
        """
        Load a pre-trained ensemble model.
        """
        self.ensemble_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

# Example usage:
# If you want to train the model:
# ensemble = EnsembleModel()
# ensemble.train(X_train, y_train)

# If you want to make predictions with a trained model:
# ensemble.load_model()
# predictions = ensemble.predict(X_test)
# probabilities = ensemble.predict_proba(X_test)
