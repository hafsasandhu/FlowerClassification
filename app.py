import os
from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from ImageClassification import OptiSA, device
import requests
import gdown

app = Flask(__name__)

# Function to download model if not exists
def download_model():
    model_path = '/tmp/best_opti_sa.pth'
    if not os.path.exists(model_path):
        model_url = "https://drive.google.com/file/d/1PKqs1vZ90QOWkMftPzeDagYA_rOcMRt2/view?usp=sharing"
        gdown.download(model_url, model_path, quiet=False)
    return model_path

# Load the model
model = OptiSA(num_classes=5).to(device)
model_path = download_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Save the uploaded file temporarily
        temp_path = '/tmp/temp_image.jpg'  # Use /tmp directory for Vercel
        file.save(temp_path)
        
        try:
            # Make prediction
            predicted_class, confidence = predict_image(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': f'{confidence:.2%}'
            })
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)})

# For local development
if __name__ == '__main__':
    app.run(debug=True) 