from flask import Flask, request, render_template, redirect, url_for
import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
from vehicle_classifier import classify_vehicle  # Import the updated classification function

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the YOLOv5 model for object detection
model = YOLO("yolov5s.pt")  # Load a pre-trained YOLOv5 model

# Route for the homepage (image upload)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle the image upload and model selection
        image = request.files.get("image")
        model_choice = request.form.get("model")  # Get the model choice from the form
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            return redirect(url_for("results", filename=image.filename, model_choice=model_choice))
        else:
            return "Error: No image uploaded.", 400
    return render_template("index.html")

# Route for displaying the results
@app.route("/results/<filename>")
def results(filename):
    model_choice = request.args.get("model_choice")  # Retrieve the selected model choice
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    # Check if the image file exists
    if not os.path.exists(image_path):
        return "Error: The specified file was not found.", 404

    # Perform object detection with YOLOv5
    results = model(image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Define the class IDs for vehicles (based on COCO dataset classes)
    vehicle_class_ids = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

    # Draw bounding boxes and labels, and crop vehicle images
    vehicle_images = []
    label_positions = []  # To store labels for each vehicle
    label_counter = 1  # Counter to label vehicles

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) in vehicle_class_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Add a label to the bounding box
                label_text = f"Vehicle {label_counter}"
                draw.text((x1, y1 - 10), label_text, fill="red")
                label_positions.append(label_text)

                # Crop the vehicle image and append to the list
                cropped_image = img.crop((x1, y1, x2, y2))
                vehicle_images.append(cropped_image)
                label_counter += 1

    # Save the image with bounding boxes and labels
    img.save(results_path)

    # Classify each cropped vehicle image using the selected model
    approximations = [
        classify_vehicle(vehicle_image, model_choice) for vehicle_image in vehicle_images
    ]

    return render_template(
        "results.html",
        filename=filename,
        approximations=approximations,
        label_positions=label_positions,
        zip=zip
    )

if __name__ == "__main__":
    app.run(debug=True)
