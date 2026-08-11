import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from multimodal_bird_species_identifier.data.image_dataset import get_image_transforms, load_image_dataset
from multimodal_bird_species_identifier.models.vision_model import BirdVisionResNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Vision model on device: {device}")

    train_tf, val_tf = get_image_transforms()
    full_dataset = load_image_dataset("data/raw/images", transform=train_tf)
    
    # Train / Validation Split (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load ResNet-18 model & optimize classifier parameters
    model = BirdVisionResNet(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.resnet.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15
    print("\n--- Starting Vision Training (ResNet-18) ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save trained vision weights
    torch.save(model.state_dict(), "data/vision_model.pt")
    print("\nVision model successfully saved to 'data/vision_model.pt'!")

if __name__ == "__main__":
    train()