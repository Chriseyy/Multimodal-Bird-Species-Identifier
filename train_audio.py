import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from multimodal_bird_species_identifier.data.dataset import BirdAudioDataset
from multimodal_bird_species_identifier.models.audio_model import BirdAudioCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Audio model on device: {device}")

    # Load dataset & split into Train / Validation sets (80% / 20%)
    dataset = BirdAudioDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = BirdAudioCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    print("\n--- Starting Audio Training ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * specs.size(0)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save trained audio weights
    torch.save(model.state_dict(), "data/audio_model.pt")
    print("\nAudio model successfully saved to 'data/audio_model.pt'!")

if __name__ == "__main__":
    train()