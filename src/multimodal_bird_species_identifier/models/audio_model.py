import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

class BirdAudioCNN(nn.Module):
    """AST (Audio Spectrogram Transformer) fine-tuned for 10 bird species."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            attn_implementation="sdpa"
        )
        
    def forward(self, input_values):
        # Remove extra channel dimension if it comes from the DataLoader
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)
            
        outputs = self.model(input_values=input_values)
        return outputs.logits