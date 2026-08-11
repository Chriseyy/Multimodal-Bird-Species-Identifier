import torch
import torch.nn as nn

from multimodal_bird_species_identifier.models.audio_model import BirdAudioCNN
from multimodal_bird_species_identifier.models.vision_model import BirdVisionResNet

class MultimodalBirdIdentifier(nn.Module):
    """Deep Feature Fusion: Combines AST (768 features) and ResNet (512 features)."""
    def __init__(self, num_classes=10, audio_weights_path=None, vision_weights_path=None):
        super().__init__()
        
        # 1. Load base submodels
        self.audio_submodel = BirdAudioCNN(num_classes=num_classes)
        self.vision_submodel = BirdVisionResNet(num_classes=num_classes, freeze_backbone=True)
        
        # 2. Load pretrained weights
        if audio_weights_path and torch.cuda.is_available():
            self.audio_submodel.load_state_dict(torch.load(audio_weights_path))
        elif audio_weights_path:
            self.audio_submodel.load_state_dict(torch.load(audio_weights_path, map_location='cpu'))

        if vision_weights_path and torch.cuda.is_available():
            self.vision_submodel.load_state_dict(torch.load(vision_weights_path))
        elif vision_weights_path:
            self.vision_submodel.load_state_dict(torch.load(vision_weights_path, map_location='cpu'))

        # 3. Freeze base submodels (train only the fusion layer)
        for param in self.audio_submodel.parameters():
            param.requires_grad = False
        for param in self.vision_submodel.parameters():
            param.requires_grad = False

        # 4. Strip classification heads (outputs are now raw embeddings)
        # AST output: 768 dimensions, ResNet output: 512 dimensions
        self.audio_submodel.model.classifier.dense = nn.Identity()
        self.vision_submodel.resnet.fc = nn.Identity()

        # 5. Define fusion classifier network
        self.fusion_classifier = nn.Sequential(
            nn.Linear(768 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_features, image_tensor):
        # Extract feature embeddings
        audio_embeds = self.audio_submodel(audio_features)  # Shape: (Batch, 768)
        vision_embeds = self.vision_submodel(image_tensor)  # Shape: (Batch, 512)
        
        # Concatenate feature embeddings -> Shape: (Batch, 1280)
        combined_features = torch.cat((audio_embeds, vision_embeds), dim=1)
        
        # Pass combined representations through fusion classifier
        return self.fusion_classifier(combined_features)