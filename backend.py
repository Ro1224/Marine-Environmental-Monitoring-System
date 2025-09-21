import os
import logging
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter
from io import BytesIO
import base64
from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define model paths
oil_spill_model_path = "oil_spill.pth"
plastic_model_path = "best.pt"  # Path to your YOLO model for plastic detection

# Load the models
model = None
plastic_model = YOLO(plastic_model_path)

if not os.path.exists(oil_spill_model_path):
    logging.error(f'Model file "{oil_spill_model_path}" not found.')
else:
    try:
        # Oil spill model setup
        class OilSpillModel(nn.Module):
            def __init__(self):
                super(OilSpillModel, self).__init__()
                self.fc1 = nn.Linear(64 * 64 * 3, 128)  # Input size depends on your image size and channels
                self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification

            def forward(self, x):
                x = x.view(-1, 64 * 64 * 3)  # Flatten the image tensor
                x = torch.relu(self.fc1(x))  # Hidden layer
                x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification output
                return x

        model = OilSpillModel()
        model.load_state_dict(torch.load(oil_spill_model_path, map_location=torch.device('cpu')))
        model.eval()
        logging.info("Oil spill model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading the oil spill model: {e}")

# Image preprocessing function
def preprocess_image(file):
    try:
        img_bytes = BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)

        # Generate preprocessed images
        grayscale_image = img.convert('L')
        edges_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
        blurred_image = grayscale_image.filter(ImageFilter.GaussianBlur(2))

        # Convert images to base64
        grayscale_image_base64 = convert_image_to_base64(grayscale_image)
        edges_image_base64 = convert_image_to_base64(edges_image)
        blurred_image_base64 = convert_image_to_base64(blurred_image)

        logging.info("Image preprocessing completed successfully.")
        return img, img_tensor, {
            "gray_image": grayscale_image_base64,
            "edges_image": edges_image_base64,
            "blurred_image": blurred_image_base64
        }
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise

# Convert image to base64 for display
def convert_image_to_base64(img):
    try:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    except Exception as e:
        logging.error(f"Error converting image to base64: {e}")
        raise

# Predict oil spill
@app.route('/predict', methods=['POST'])
def oil_spill_detection():
    try:
        if model is None:
            logging.error("Model is not loaded.")
            return jsonify({"error": "Model is not loaded. Check logs for details."}), 500

        file = request.files.get('file')
        if not file:
            logging.error("No file provided in the request.")
            return jsonify({"error": "No file provided."}), 400

        img, img_tensor, preprocessed_images = preprocess_image(file)

        # Perform prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        probability_of_spill = output.item()
        threshold = 0.7

        if probability_of_spill >= threshold:
            predicted_class = "oil_spill"
            spill_percentage = probability_of_spill * 100
        else:
            predicted_class = "no_oil_spill"
            spill_percentage = (1 - probability_of_spill) * 100

        # Zone classification
        if spill_percentage >= 75:
            zone = "High-Risk Zone"
        elif spill_percentage >= 50:
            zone = "Medium-Risk Zone"
        else:
            zone = "Low-Risk Zone"

        logging.info(f"Prediction: {predicted_class}, Spill Percentage: {spill_percentage:.2f}%, Zone: {zone}")

        uploaded_image_base64 = convert_image_to_base64(img)

        return jsonify({
            "result": predicted_class,
            "spill_percentage": spill_percentage,
            "threshold": threshold,
            "zone": zone,
            "uploaded_image": uploaded_image_base64,
            "gray_image": preprocessed_images["gray_image"],
            "edges_image": preprocessed_images["edges_image"],
            "blurred_image": preprocessed_images["blurred_image"]
        })
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# Predict plastic waste
@app.route('/predict_plastic', methods=['POST'])
def predict_plastic():
    try:
        # Retrieve the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided in the request."}), 400
        
        # Process the image with YOLO for plastic detection
        img = Image.open(file)
        img = img.convert("RGB")  # Ensure compatibility with YOLO
        
        # Perform prediction
        results = plastic_model.predict(img)
        
        # Check if results contain valid detections
        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "prediction": "No Plastic Detected",
                "result_image": None
            })
        
        # Plot BBoxes on the image
        annotated_img = results[0].plot(line_width=1, conf=0.25)[:, :, ::-1]  # Convert BGR to RGB for PIL
        annotated_img_pil = Image.fromarray(annotated_img)
        
        # Convert annotated image to base64 for rendering
        buffered = BytesIO()
        annotated_img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Extract predicted labels and confidence
        LABELS = {
            0: 'PLASTIC_BAG',
            1: 'PLASTIC_BOTTLE',
            2: 'OTHER_PLASTIC_WASTE',
            3: 'NOT_PLASTIC_WASTE'
        }
        
        predictions = []
        confidences = []
        for box in results[0].boxes.data.tolist():
            label_index = int(box[5])  # Ensure the label index exists
            confidence = float(box[4])
            predictions.append(LABELS.get(label_index, "Unknown"))
            confidences.append(confidence)
        
        if not predictions:
            return jsonify({
                "prediction": "No Plastic Detected",
                "result_image": img_str
            })
        
        # Format the output with labels and confidence
        formatted_predictions = []
        for label, confidence in zip(predictions, confidences):
            formatted_predictions.append(f"{label} ({confidence:.2f})")
        
        return jsonify({
            "prediction": " | ".join(formatted_predictions),
            "result_image": img_str
        })
        
    except Exception as e:
        logging.error(f"Error in predict_plastic: {e}")
        return jsonify({"error": str(e)}), 500

# Render pages
@app.route('/index', methods=['GET'])
def index():
    return render_template('oil_spill.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/modules', methods=['GET'])
def modules():
    return render_template('modules.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/indexs', methods=['GET', 'POST'])
def indexs():
    if request.method == 'POST':
        # Handle POST request logic for plastic detection
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided."}), 400
        
        # Process the image with YOLO for plastic detection
        img = Image.open(file)
        img = img.convert("RGB")  # Ensure compatibility with YOLO
        
        # Perform prediction
        results = plastic_model.predict(img)
        
        # Check if results contain valid detections
        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "prediction": "No Plastic Detected",
                "result_image": None
            })
        
        # Plot BBoxes on the image
        annotated_img = results[0].plot(line_width=1, conf=0.25)[:, :, ::-1]  # Convert BGR to RGB for PIL
        annotated_img_pil = Image.fromarray(annotated_img)
        
        # Convert annotated image to base64 for rendering
        buffered = BytesIO()
        annotated_img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Extract predicted labels and confidence
        LABELS = {
            0: 'PLASTIC_BAG',
            1: 'PLASTIC_BOTTLE',
            2: 'OTHER_PLASTIC_WASTE',
            3: 'NOT_PLASTIC_WASTE'
        }
        
        predictions = []
        confidences = []
        for box in results[0].boxes.data.tolist():
            label_index = int(box[5])  # Ensure the label index exists
            confidence = float(box[4])
            predictions.append(LABELS.get(label_index, "Unknown"))
            confidences.append(confidence)
        
        if not predictions:
            return jsonify({
                "prediction": "No Plastic Detected",
                "result_image": img_str
            })
        
        # Format the output with labels and confidence
        formatted_predictions = []
        for label, confidence in zip(predictions, confidences):
            formatted_predictions.append(f"{label} ({confidence:.2f})")
        
        return jsonify({
            "prediction": " | ".join(formatted_predictions),
            "result_image": img_str
        })
    else:
        return render_template('plastic_detection.html')

if __name__ == "__main__":
    app.run(debug=True)
