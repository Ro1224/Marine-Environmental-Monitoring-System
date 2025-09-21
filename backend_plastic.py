import os
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from io import BytesIO

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")


def process_image(file):
    """
    Process the uploaded image file with the YOLO model.
    Returns the prediction details and the annotated image.
    """
    try:
        # Open the uploaded file
        img = Image.open(file)
        img = img.convert("RGB")  # Ensure compatibility with YOLO

        # Perform prediction
        results = model.predict(img)

        # Check if results contain valid detections
        if not results or len(results[0].boxes) == 0:
            return {
                "prediction": "No Plastic Detected",
                "result_image": None
            }

        # Plot BBoxes on the image
        annotated_img = results[0].plot(line_width=1, conf=0.25)[:, :, ::-1]  # Convert BGR to RGB for PIL
        annotated_img_pil = Image.fromarray(annotated_img)

        # Convert annotated image to base64
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
            return {
                "prediction": "No Plastic Detected",
                "result_image": img_str
            }

        # Format the output with labels and confidence
        formatted_predictions = []
        for label, confidence in zip(predictions, confidences):
            formatted_predictions.append(f"{label} ({confidence:.2f})")

        # Return the most common prediction and formatted predictions
        return {
            "prediction": " | ".join(formatted_predictions),
            "result_image": img_str
        }

    except Exception as e:
        # Print the error for debugging
        print("Error in process_image:", str(e))
        return {"error": str(e)}


@app.route('/')
def index():
    """
    Renders the HTML page with the upload form.
    """
    return render_template('plastic_detection.html')


@app.route('/indexs', methods=['POST'])
def predict():
    """
    Handles the POST request, processes the image, and returns the result.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Process the image
        result = process_image(file)

        if "error" in result:
            return jsonify(result), 500

        # Return JSON response
        return jsonify(result)

    return jsonify({"error": "Invalid request"}), 400


if __name__ == "__main__":
    app.run(debug=True)
