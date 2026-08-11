import torch
import torch.nn as nn

class BirdAudioCNN(nn.Module):
    """Lightweight 2D-CNN for audio Mel-spectrogram classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


if __name__ == "__main__":
    model = BirdAudioCNN(num_classes=10)
    dummy_audio = torch.randn(4, 1, 128, 130)
    output = model(dummy_audio)
    
    print("Audio model loaded successfully!")
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape (logits): {output.shape}")