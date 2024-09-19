from flask import Flask, request, render_template, redirect
from PIL import Image
import torch
from torchvision import transforms
from model import build_model
import os, csv

app = Flask(__name__)

# Initialize and load the model
net = build_model(num_classes=196)
net.load_state_dict(torch.load('outputs/model.pth'))
net.eval()

# Image transformation setup
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to read class names
def get_class_names(csv_path):
    with open(csv_path, 'r') as f:
        return [row[0] for row in csv.reader(f)]

# Load the car model class names
car_classes = get_class_names('../input/names.csv')

# Prediction logic
def make_prediction(image_file):
    img = Image.open(image_file).convert('RGB')
    img_tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        predictions = torch.nn.functional.softmax(net(img_tensor), dim=1)
        top_5 = predictions.topk(5)
    return [(car_classes[idx], prob.item()) for idx, prob in zip(top_5.indices[0], top_5.values[0])]

# Route to handle file uploads and display predictions
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        img_file = request.files['file']
        save_path = os.path.join(app.root_path, 'static/uploads', img_file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_file.save(save_path)
        preds = make_prediction(save_path)
        return render_template('result.html', predictions=preds, image_path=f'static/uploads/{img_file.filename}')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
