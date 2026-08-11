import torch
import torch.nn as nn

from multimodal_bird_species_identifier.models.audio_model import BirdAudioCNN
from multimodal_bird_species_identifier.models.vision_model import BirdVisionResNet

class MultimodalBirdIdentifier(nn.Module):
    """Multimodal model combining Audio CNN and Vision ResNet via late feature fusion."""
    def __init__(self, num_classes=10, audio_weights_path=None, vision_weights_path=None):
        super().__init__()
        
        # 1. Base Models
        self.audio_submodel = BirdAudioCNN(num_classes=num_classes)
        self.vision_submodel = BirdVisionResNet(num_classes=num_classes, freeze_backbone=True)
        
        # Load pre-trained weights if provided
        if audio_weights_path:
            self.audio_submodel.load_state_dict(torch.load(audio_weights_path, map_location='cpu'))
        if vision_weights_path:
            self.vision_submodel.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))
            
        # Strip final classification layers to extract raw feature vectors (256-dim each)
        self.audio_submodel.classifier = self.audio_submodel.classifier[:-1]
        self.vision_submodel.resnet.fc = self.vision_submodel.resnet.fc[:-1]
        
        # 2. Joint Multimodal Fusion Classifier (256 + 256 = 512 input features)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_spec, image_tensor):
        # Extract 256-dim feature embeddings
        audio_features = self.audio_submodel(audio_spec)
        vision_features = self.vision_submodel(image_tensor)
        
        # Concatenate along feature dimension -> Shape: (Batch, 512)
        combined_features = torch.cat((audio_features, vision_features), dim=1)
        
        # Output final species logits
        return self.fusion_classifier(combined_features)


if __name__ == "__main__":
    # Test fusion model instantiation and forward pass
    fusion_model = MultimodalBirdIdentifier(num_classes=10)
    
    dummy_audio = torch.randn(2, 1, 128, 130)
    dummy_image = torch.randn(2, 3, 224, 224)
    
    output_logits = fusion_model(dummy_audio, dummy_image)
    
    print("Multimodal Fusion Model initialized successfully!")
    print(f"Combined Output Shape (Logits): {output_logits.shape}")