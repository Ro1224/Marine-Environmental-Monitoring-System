# Marine Environmental Monitoring System

## Overview
This project is a comprehensive web-based system for detecting and analyzing marine environmental hazards, specifically focusing on oil spills and plastic waste detection in marine environments. The system utilizes deep learning models and computer vision techniques to provide real-time analysis and classification of environmental threats.

## Features

### 1. Oil Spill Detection
- Real-time analysis of oil spill presence in marine images
- Risk zone classification (High-Risk, Medium-Risk, Low-Risk)
- Probability assessment of oil spill presence
- Advanced image processing with multiple visualization modes:
  - Grayscale analysis
  - Edge detection
  - Gaussian blur filtering

### 2. Plastic Waste Detection
- YOLO-based object detection for marine plastic waste
- Multiple plastic waste classification categories:
  - Plastic bags
  - Plastic bottles
  - Other plastic waste
  - Non-plastic waste
- Confidence scoring for detected items
- Visual bounding box annotations on detected objects

### 3. Cleanup Methods Analysis
- Integration with marine cleanup strategies
- Real-time fetching of cleanup methods from marine information sources
- Detailed descriptions of various cleanup approaches
- Interactive presentation of cleanup strategies

## Technical Stack

### Backend
- Flask (Python web framework)
- PyTorch (Deep Learning framework)
- YOLO (You Only Look Once) for object detection
- OpenCV and PIL for image processing
- BeautifulSoup for web scraping

### Frontend
- HTML/CSS templates
- JavaScript for interactive features
- Bootstrap for responsive design

### Models
- Custom CNN model for oil spill detection
- YOLO model for plastic waste detection
- Pre-trained weights included:
  - `oil_spill.pth`
  - `best.pt` (YOLO model)

## Project Structure
```
├── app.py                      # Flask application entry point
├── backend.py                  # Main backend logic
├── backend_oilspill.py        # Oil spill detection specific backend
├── backend_plastic.py         # Plastic detection specific backend
├── models/                    # Model directory
│   └── oil_spill.h5          # Oil spill detection model
├── static/                    # Static files (images, CSS, JS)
│   └── images/               # Image assets
├── templates/                 # HTML templates
│   ├── index.html            # Home page
│   ├── modules.html          # Modules overview
│   ├── oil_spill.html        # Oil spill detection interface
│   ├── plastic_detection.html # Plastic detection interface
│   └── ...                   # Other template files
└── uploads/                  # Directory for uploaded images
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure model files are in place:
- Place `oil_spill.pth` in the root directory
- Place `best.pt` (YOLO model) in the root directory

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Access the web interface:
- Open a web browser and navigate to `http://localhost:5000`
- Select the desired analysis module (Oil Spill or Plastic Detection)
- Upload an image for analysis
- View the results and analysis

## Features in Detail

### Oil Spill Detection
- Uploads marine images for analysis
- Processes images through multiple visualization modes
- Provides spill probability percentage
- Classifies risk zones based on detection confidence
- Generates detailed analysis reports

### Plastic Detection
- Real-time object detection in marine images
- Multiple category classification
- Confidence scores for each detection
- Visual annotations of detected objects
- Detailed classification reports

## Contributing
Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

