import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from diffusion import UNet
from ViT import EdgeAwareViT

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
vit = EdgeAwareViT().to(device)  # Ensure the model is properly initialized
unet = UNet().to(device)

# Load trained weights
vit.load_state_dict(torch.load("../models/ea_vit_model.pth", map_location=device))
unet.load_state_dict(torch.load("../models/unet_model.pth", map_location=device))

# Set models to evaluation mode
vit.eval()
unet.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load a sample image from dataset
sample_image_path = "../test_data/test2.png"  # Change path based on dataset structure
image = Image.open(sample_image_path).convert("RGB")

# Apply transforms
input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Resize to 48x48 for U-Net
small_image = torch.nn.functional.interpolate(input_image, size=(48, 48), mode="bilinear", align_corners=False)

# Enhance using U-Net
with torch.no_grad():
    enhanced_image = unet(small_image)

# Resize back to 224x224 for ViT
enhanced_image = torch.nn.functional.interpolate(enhanced_image, size=(224, 224), mode="bilinear", align_corners=False)

# Ensure the enhanced image has 3 channels for ViT (if needed)
if enhanced_image.shape[1] == 1:  # If grayscale, convert to RGB format
    enhanced_image = enhanced_image.repeat(1, 3, 1, 1)

# Predict with ViT
with torch.no_grad():
    output = vit(enhanced_image)
    predicted_class = torch.argmax(output, dim=1).item()

# Display result
plt.imshow(image)
plt.title(f"Predicted Class: {predicted_class}")
plt.axis("off")
plt.show()
