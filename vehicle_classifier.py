import torch
from PIL import Image
from torchvision import transforms, models
import pandas as pd
from efficientnet_pytorch import EfficientNet

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load car labels for Stanford Cars
stanford_car_names = pd.read_csv('stanford_names.csv', header=None)
stanford_car_names.columns = ["Model"]
stanford_labels = stanford_car_names["Model"].sort_values().tolist()

# Load car labels for Kaggle Cars
kaggle_car_names = pd.read_csv('kaggle_names.csv').set_index('Label').to_dict()['Model']

# Load the EfficientNet-B3 model for Stanford Cars
car_model_classifier_stanford = models.efficientnet_b3()
num_features = car_model_classifier_stanford.classifier[1].in_features
car_model_classifier_stanford.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 196)
)
car_model_classifier_stanford.load_state_dict(
    torch.load("best_stanford_cars_model.pth", map_location=device)
)
car_model_classifier_stanford = car_model_classifier_stanford.to(device)
car_model_classifier_stanford.eval()

# Load the EfficientNet-B0 model for Kaggle Cars
car_model_classifier_kaggle = EfficientNet.from_name('efficientnet-b0')
car_model_classifier_kaggle._fc = torch.nn.Linear(
    car_model_classifier_kaggle._fc.in_features, len(kaggle_car_names)
)
car_model_classifier_kaggle.load_state_dict(
    torch.load("vehicle_classifier.pth", map_location='cpu')  # Load on CPU, same as iLab
)
car_model_classifier_kaggle = car_model_classifier_kaggle.to(device)
car_model_classifier_kaggle.eval()

# Define image transformations for both models
transform_stanford = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_kaggle = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to classify vehicle images
def classify_vehicle(image, model_choice):
    # Use different transformations based on the model
    if model_choice == "stanford":
        model = car_model_classifier_stanford
        labels = stanford_labels
        image_tensor = transform_stanford(image).unsqueeze(0).to(device)
        use_dict = False  # Indicates we are using a list for labels
    else:
        model = car_model_classifier_kaggle
        labels = kaggle_car_names
        image_tensor = transform_kaggle(image).unsqueeze(0).to(device)
        use_dict = True  # Indicates we are using a dictionary for labels

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_probabilities, top5_indices = probabilities.topk(5, 1, largest=True, sorted=True)

        # Handle label retrieval based on the type of labels (list or dictionary)
        if use_dict:
            predictions = [labels.get(idx.item(), "Unknown") for idx in top5_indices[0]]
        else:
            predictions = [labels[idx.item()] if idx.item() < len(labels) else "Unknown" for idx in top5_indices[0]]

    return predictions
