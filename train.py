import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as optim

import numpy as np
import random
from perlin_noise import PerlinNoise

from diffusion import UNet
from ViT import EdgeAwareViT

#---------------------------------------------------------diffusion model------------------------------------------------------------------------


unet = UNet()

def generate_cloud_noise(image, scale=10, intensity=0.5):
    C, H, W = image.shape  

    perlin = PerlinNoise(octaves=scale)
    
    noise = np.array([[perlin([i / H, j / W]) for j in range(W)] for i in range(H)])
    noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize [0,1]

    cloud_noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

    cloudy_image = image * (1 - intensity) + cloud_noise * intensity
    return cloudy_image.clamp(0, 1)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)

transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_root = "../data"
dataset = datasets.ImageFolder(root=data_root, transform=transforms)

batch_size = 4
indices = random.sample(range(len(dataset)), 8000)
subset = Subset(dataset, indices)

train_size = int(0.8 * len(subset))  
val_size = len(subset) - train_size  
train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

epochs = 3
log_file = "unet_training_log.txt"

with open(log_file, "w") as f:
    
    for epoch in range(epochs):      
        f.write(f"Epoch : {epoch}\n")
        unet.train()
        train_loss = 0
        counter = 0
        
        for images, _ in train_loader:
            print(counter)  
            images = images.to('cpu')
            cloudy_images = torch.stack([generate_cloud_noise(img) for img in images]).to('cpu')

            optimizer.zero_grad()
            outputs = unet(cloudy_images)

            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            counter += 1
        
        avg_train_loss = train_loss / counter
        f.write(f"Train Loss: {avg_train_loss:.4f}\n")

        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to('cpu')
                cloudy_images = torch.stack([generate_cloud_noise(img) for img in images]).to('cpu')

                outputs = unet(cloudy_images)
                loss = loss_fn(outputs, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n\n")

torch.save(unet.state_dict(), "../models/unet_model.pth")
print("Model saved successfully!")

#---------------------------------------------------------edge aware ViT------------------------------------------------------------------------

vit = EdgeAwareViT(num_classes=10, pretrained_path="vit_model.pth")

transforms_vit = transforms.Compose([
    transforms.Resize((224, 224)),  # Required for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_root = "./data"
dataset_vit = datasets.ImageFolder(root=data_root, transform=transforms_vit)

class_indices = {}
for idx, (_, label) in enumerate(dataset_vit.samples):
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)

num_classes = len(class_indices)
samples_per_class = 16000 // num_classes  # Distribute samples evenly

selected_indices = []
for label, indices in class_indices.items():
    selected_indices.extend(random.sample(indices, min(samples_per_class, len(indices))))

subset_vit = Subset(dataset_vit, selected_indices)

train_size = int(0.8 * len(subset_vit))  
val_size = len(subset_vit) - train_size  
train_dataset_vit, val_dataset_vit = random_split(subset_vit, [train_size, val_size])

batch_size = 4
train_loader_vit = DataLoader(train_dataset_vit, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader_vit = DataLoader(val_dataset_vit, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


device = 'cpu'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(), lr=0.0001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit.to(device)
unet.to(device)  

log_file = "training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch, Train Loss, Validation Loss, Accuracy\n")

for epoch in range(num_epochs):
    vit.train()
    running_loss = 0.0
    counter = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        print(counter)
        # Resize images to 48x48 for U-Net
        small_images = F.interpolate(images, size=(48, 48), mode="bilinear", align_corners=False)
        
        # Enhance using U-Net
        with torch.no_grad():
            enhanced_images = unet(small_images)

        # Resize back to 224x224 for ViT
        enhanced_images = F.interpolate(enhanced_images, size=(224, 224), mode="bilinear", align_corners=False)
        optimizer.zero_grad()
        outputs = vit(enhanced_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        counter+=1

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    vit.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            small_images = F.interpolate(images, size=(48, 48), mode="bilinear", align_corners=False)
            enhanced_images = unet(small_images)
            enhanced_images = F.interpolate(enhanced_images, size=(224, 224), mode="bilinear", align_corners=False)

            outputs = vit(enhanced_images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1}, {avg_train_loss:.4f}, {avg_val_loss:.4f}, {accuracy:.2f}%\n")

torch.save(vit.state_dict(), "../models/ea_vit.pth")
print("Edge Aware ViT model saved successfully!")