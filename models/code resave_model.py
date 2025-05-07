import torch
import torchvision.models as models

# Load a sample pretrained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)

# Save it to your backend/models directory
torch.save(model, 'backend/models/visual_cnn_model.pt')
