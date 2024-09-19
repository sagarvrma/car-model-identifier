import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets import initialize_loaders
from model import build_model
from tqdm import tqdm
import os

def execute_training(model, criterion, optimizer, scheduler, epochs=25):
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            phase_loss, phase_correct = 0.0, 0
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                phase_loss += loss.item() * inputs.size(0)
                phase_correct += torch.sum(predictions == labels.data)

            loss_avg = phase_loss / dataset_sizes[phase]
            accuracy = phase_correct.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {loss_avg:.4f} | Accuracy: {accuracy:.4f}')
        scheduler.step()
    return model

if __name__ == "__main__":
    train_dir = '../input/car_data/train'
    test_dir = '../input/car_data/test'
    
    # Initialize data loaders
    train_loader, test_loader = initialize_loaders(train_dir, test_dir, batch_size=32)
    dataloaders = {'train': train_loader, 'test': test_loader}
    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'test']}
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model and optimization setup
    model = build_model(num_classes=196).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Save directory setup
    output_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, 'model.pth')

    print(f"Model will be saved to {output_model_path}.")
    if input("Proceed with training? (yes/no): ").lower() == 'yes':
        model = execute_training(model, criterion, optimizer, scheduler, epochs=25)
        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved at {output_model_path}")
    else:
        print("Training aborted.")
