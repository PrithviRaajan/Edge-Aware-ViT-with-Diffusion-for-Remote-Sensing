import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16
import cv2


vit_model = vit_b_16(pretrained=True)

torch.save(vit_model.state_dict(), "vit_model.pth")
print("ViT model saved successfully!")

def extract_edges(image_tensor):
    image_np = image_tensor.permute(1,2,0).cpu().numpy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edges = torch.tensor(edges).unsqueeze(0)
    return edges

class EdgeAwareViT(nn.Module):
    def __init__(self, num_classes=10, pretrained_path="vit_model.pth"):
        super(EdgeAwareViT, self).__init__()

        self.vit = vit_b_16(pretrained=False)
        
        self.vit.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        print(f"Loaded ViT weights from {pretrained_path}")

        self.edge_embed = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        self.vit.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        edge_maps = torch.stack([extract_edges(img) for img in x], dim=0).to(x.device).float()
        edge_features = self.edge_embed(edge_maps)
        x = x + edge_features
        return self.vit(x)
