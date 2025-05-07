# utils/ml_pipeline_optimizer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class MLPipelineOptimizer:
    def __init__(self, data, target_col, model_type='url'):
        self.data = data
        self.target_col = target_col
        self.model_type = model_type

    def preprocess(self):
        """Splits data into train and test sets"""
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_pipeline(self):
        """Creates a scikit-learn pipeline"""
        if self.model_type == 'text':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression())
            ])
            param_grid = {
                'tfidf__max_df': [0.7, 0.85, 1.0],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__C': [0.1, 1, 10]
            }
        else:  # URL model
            pipeline = Pipeline([
                ('clf', RandomForestClassifier())
            ])
            param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5]
            }

        return pipeline, param_grid

    def run_grid_search(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        pipeline, param_grid = self.build_pipeline()

        grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid.fit(X_train.squeeze(), y_train)

        print("Best Params:", grid.best_params_)
        print("Classification Report:\n", classification_report(y_test, grid.predict(X_test.squeeze())))

        return grid.best_estimator_

    def save_model(self, model_path, vectorizer_path=None):
        """Saves the model (and optionally the vectorizer)"""
        joblib.dump(self.best_model, model_path)
        if vectorizer_path and hasattr(self.best_model, 'named_steps'):
            tfidf = self.best_model.named_steps.get('tfidf')
            if tfidf:
                joblib.dump(tfidf, vectorizer_path)

    def optimize_and_save(self, model_path, vectorizer_path=None):
        self.best_model = self.run_grid_search()
        self.save_model(model_path, vectorizer_path)
