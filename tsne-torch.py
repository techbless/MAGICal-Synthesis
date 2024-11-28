import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and transform dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming your dataset is organized as: root/directory_1, root/directory_2, ..., root/directory_n
# Replace 'root_directory' with the path to your dataset root
dataset = ImageFolder(root='./Result10', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model and modify it for feature extraction
model = resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Modify the last layer to output features directly
model = model.to(device)
model.eval()

# Extract features
features = []
labels = []
with torch.no_grad():
    for inputs, label in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        features.append(outputs.cpu().numpy())
        labels.append(label.numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# Plot
plt.figure(figsize=(10, 10))
for i in np.unique(labels):
    indices = labels == i
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=dataset.classes[i])
plt.legend()
plt.show()
