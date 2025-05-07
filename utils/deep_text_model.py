# utils/deep_text_model.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import Adam
from tqdm import tqdm


class PhishingTextDataset(Dataset):
    """Custom Dataset to load text data for BERT model."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_deep_text_model(csv_path, model_save_path):
    """Trains a deep learning model (BERT) for phishing text detection."""
    # Load data
    df = pd.read_csv(csv_path)
    texts = df['text'].values
    labels = df['label'].values

    # Tokenizer for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = PhishingTextDataset(X_train, y_train, tokenizer)
    test_dataset = PhishingTextDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize BERT model for classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Train model
    model.train()
    for epoch in range(3):  # Run for 3 epochs
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in loop:
            optimizer.zero_grad()
            
            # Get input IDs, attention masks and labels from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # Evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Model Evaluation:\n", classification_report(all_labels, all_preds))

    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"✅ Model saved to {model_save_path}")


def predict_text(text, model_path):
    """Uses a trained BERT model to predict phishing intent in text."""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Prepare input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

    return preds.item()


# Specify the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

