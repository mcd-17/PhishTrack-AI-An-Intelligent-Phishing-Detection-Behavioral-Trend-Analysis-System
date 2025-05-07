import torch
from torchvision import transforms
from PIL import Image
import os
import logging
import torch.nn as nn

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the CNN Model
class VisualCNNModel(nn.Module):
    def __init__(self):
        super(VisualCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Fixed dimension after pooling for 128x128 input
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
def load_model(model_path="models/visual_cnn_model.pt"):
    model = VisualCNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded successfully.")
    return model

# Preprocess an input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

# Predict using the model
def predict(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()

# Evaluate a folder of images
def evaluate_images_from_directory(dataset_path="utils/phishing+websites"):
    model = load_model()
    correct = 0
    total = 0

    for label in ['phishing', 'legitimate']:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith((".png", ".jpg", ".jpeg")):
                total += 1
                pred = predict(os.path.join(folder, file), model)
                if pred == (0 if label == 'phishing' else 1):
                    correct += 1

    accuracy = 100 * correct / total if total > 0 else 0
    logging.info(f"Accuracy: {accuracy:.2f}% on {total} samples.")

# Run evaluation if this script is the entry point
if __name__ == "__main__":
    evaluate_images_from_directory()
