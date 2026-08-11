import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BirdAudioCNN(nn.Module):
    """ResNet-18 backbone adapted for 1-channel Mel-Spectrograms (Transfer Learning)."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
            
        # Classifier auf 10 Vogelarten anpassen
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = BirdAudioCNN(num_classes=10)
    dummy_audio = torch.randn(4, 1, 128, 130)  # Mel-Spektrogramm Batch
    output = model(dummy_audio)
    
    print("Pretrained Audio ResNet-18 loaded successfully!")
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape (logits): {output.shape}")