import os
import glob
import random
import torch
import torchaudio
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torchvision import transforms
from transformers import AutoFeatureExtractor

from multimodal_bird_species_identifier.models.multimodal_net import MultimodalBirdIdentifier

# ---------------------------------------------------------
# 1. MULTIMODAL DATASET (Image + Audio Pairs)
# ---------------------------------------------------------
class PairedMultimodalDataset(Dataset):
    def __init__(self, image_dir="data/raw/images", audio_dir="data/raw/audio"):
        self.classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            img_files = sorted(glob.glob(os.path.join(image_dir, cls_name, "*.jpg")))
            aud_files = sorted(glob.glob(os.path.join(audio_dir, cls_name, "*.mp3")))
            
            for img_path in img_files:
                if aud_files:
                    self.samples.append((img_path, aud_files, self.class_to_idx[cls_name]))

        # Image transformations
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # AST Audio Feature Extractor
        self.ast_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.target_sr = 16000
        self.target_samples = 163840

    def __len__(self):
        return len(self.samples)

    def process_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

            speech = waveform.squeeze(0)
            if speech.shape[0] > self.target_samples:
                speech = speech[:self.target_samples]
            else:
                speech = torch.nn.functional.pad(speech, (0, self.target_samples - speech.shape[0]))

            inputs = self.ast_extractor(speech.numpy(), sampling_rate=self.target_sr, return_tensors="pt")
            return inputs.input_values.squeeze(0)
        except Exception:
            return torch.zeros((1024, 128))

    def __getitem__(self, idx):
        img_path, aud_files, label = self.samples[idx]
        
        # 1. Load image
        img_tensor = self.img_transform(Image.open(img_path).convert("RGB"))
        
        # 2. Load random audio track of the same bird species
        aud_path = random.choice(aud_files)
        audio_tensor = self.process_audio(aud_path)
        
        return audio_tensor, img_tensor, label

# ---------------------------------------------------------
# 2. TRAINING LOOP FOR MULTIMODAL FUSION
# ---------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Multimodal Fusion on device: {device}")

    dataset = PairedMultimodalDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model and load pretrained weights for base submodels
    model = MultimodalBirdIdentifier(
        num_classes=len(dataset.classes),
        audio_weights_path="data/audio_model.pt",
        vision_weights_path="data/vision_model.pt"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # Train ONLY the fusion classifier head (fast convergence)
    optimizer = AdamW(model.fusion_classifier.parameters(), lr=1e-3)

    epochs = 5
    print("\n--- Starting Multimodal Deep Fusion Training ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for audio_inputs, images, labels in train_loader:
            audio_inputs, images, labels = audio_inputs.to(device), images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio_inputs, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for audio_inputs, images, labels in val_loader:
                audio_inputs, images, labels = audio_inputs.to(device), images.to(device), labels.to(device)
                outputs = model(audio_inputs, images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "data/multimodal_model.pt")
    print("\nMultimodal Fusion model successfully trained and saved to 'data/multimodal_model.pt'!")

if __name__ == "__main__":
    train()