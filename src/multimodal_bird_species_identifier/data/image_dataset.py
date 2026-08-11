import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_image_transforms():
    """Defines training data augmentation and validation transforms."""
    train_transforms = transforms.Compose([
        # Direct zoom-crop on raw images (skips pre-resize to avoid double interpolation)
        # scale=(0.4, 1.0) forces the network to learn zoomed-in bird features
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation pipeline that preserves aspect ratio instead of squishing images
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms

def load_image_dataset(data_dir="data/raw/images", transform=None):
    """Loads the ImageFolder dataset from the directory structure."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' was not found!")
        
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset


if __name__ == "__main__":
    train_tf, _ = get_image_transforms()
    dataset = load_image_dataset("data/raw/images", transform=train_tf)
    
    print(f"Found image classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"Total number of images: {len(dataset)}")
    
    if len(dataset) > 0:
        img_tensor, label_id = dataset[0]
        print(f"Image tensor shape (C x H x W): {img_tensor.shape}")
        print(f"Sample Label ID: {label_id} ({dataset.classes[label_id]})")