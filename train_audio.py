import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from multimodal_bird_species_identifier.data.dataset import BirdAudioDataset
from multimodal_bird_species_identifier.models.audio_model import BirdAudioCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning AST Audio Transformer on device: {device}")

    # Load dataset
    dataset = BirdAudioDataset(audio_dir="data/raw/audio")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Batch size set to 4 to fit into VRAM of most GPUs
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = BirdAudioCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Small learning rate (2e-5) is crucial for fine-tuning Transformers!
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    epochs = 10
    print("\n--- Starting AST Fine-Tuning ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation Loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save weights
    torch.save(model.state_dict(), "data/audio_model.pt")
    print("\nAST Audio model successfully fine-tuned and saved to 'data/audio_model.pt'!")

if __name__ == "__main__":
    train()