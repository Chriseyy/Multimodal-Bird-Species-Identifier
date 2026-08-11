import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BirdVisionResNet(nn.Module):
    """ResNet-18 model with pre-trained ImageNet weights for bird species classification."""
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained ResNet-18 model
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        
        # Freeze backbone parameters to speed up training
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # Replace final fully connected layer with custom classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # num_classes = 10
        )

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    model = BirdVisionResNet(num_classes=10)
    dummy_img = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_img)
    
    print("Vision model loaded successfully!")
    print(f"Input shape: {dummy_img.shape}")
    print(f"Output shape (logits): {output.shape}")