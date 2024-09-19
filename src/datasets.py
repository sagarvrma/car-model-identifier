from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def create_data_loader(data_dir, batch_size=32, shuffle=False, augment=False):
    # Apply augmentations for the training set
    if augment:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = ImageFolder(root=data_dir, transform=data_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def initialize_loaders(train_dir, test_dir, batch_size=32):
    train_loader = create_data_loader(train_dir, batch_size, shuffle=True, augment=True)
    test_loader = create_data_loader(test_dir, batch_size, shuffle=False)
    
    print(f"Training set: {len(train_loader)} batches")
    print(f"Test set: {len(test_loader)} batches")
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_path = '../input/car_data/train'
    test_path = '../input/car_data/test'

    train_loader, test_loader = initialize_loaders(train_path, test_path, batch_size=4)

    for imgs, lbls in train_loader:
        print(f'Loaded batch of images with shape: {imgs.shape}')
        print(f'Loaded batch of labels: {lbls}')
        break
