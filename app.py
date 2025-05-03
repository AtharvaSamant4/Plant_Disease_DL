import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from PIL import ImageOps
import requests
import re
import cv2
from opencage.geocoder import OpenCageGeocode
from skimage.feature import graycomatrix, graycoprops

# Configure TensorFlow for low memory usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Model paths
MODEL_PATHS = {
    "plant": "Plant_Classification_Model.tflite",
    "disease": "Plant_Disease_Predictor_with_Weather.tflite"
}

# Lazy-loaded models
_MODELS = {
    "plant": None,
    "disease": None
}

class LeafDetector:
    def __init__(self):
        self.COLOR_THRESH = 0.25
        self.CONTRAST_MAX = 450
        self.AREA_RANGE = (500, 50000)
        self.PERIMETER_RATIO = 0.25

    def detect(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        return (self._color_detection(img) + 
                self._shape_detection(img) + 
                self._texture_detection(img)) >= 2

    def _color_detection(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return cv2.countNonZero(mask) / (img.size/3) > self.COLOR_THRESH

    def _shape_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not self.AREA_RANGE[0] < area < self.AREA_RANGE[1]:
                continue
            perimeter = cv2.arcLength(cnt, True)
            compactness = (perimeter**2) / (4 * np.pi * area)
            if compactness < self.PERIMETER_RATIO:
                return True
        return False

    def _texture_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)
        return graycoprops(glcm, 'contrast')[0,0] < self.CONTRAST_MAX

def load_model(model_name):
    if _MODELS[model_name] is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS[model_name])
        interpreter.allocate_tensors()
        _MODELS[model_name] = interpreter
    return _MODELS[model_name]

def clear_memory():
    for key in _MODELS:
        _MODELS[key] = None
    tf.keras.backend.clear_session()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TOP_K = 3
HIGH_CONFIDENCE_THRESHOLD = 0.9
PLANT_CLASS_NAMES = ['Apple', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 
                    'Pepper_bell', 'Potato', 'Soybean', 'Strawberry', 'Tomato']

DISEASE_LABELS = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy', 4: 'Cherry_(including_sour)___Powdery_mildew',
    5: 'Cherry_(including_sour)___healthy', 6: 'Corn_(maize)___Common_rust_',
    7: 'Corn_(maize)___Northern_Leaf_Blight', 8: 'Corn_(maize)___healthy',
    9: 'Grape___Black_rot', 10: 'Grape___Esca_(Black_Measles)',
    11: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 12: 'Grape___healthy',
    13: 'Orange___Haunglongbing_(Citrus_greening)', 14: 'Peach___Bacterial_spot',
    15: 'Peach___healthy', 16: 'Pepper,_bell___Bacterial_spot',
    17: 'Pepper,_bell___healthy', 18: 'Potato___Early_blight',
    19: 'Potato___Late_blight', 20: 'Potato___healthy',
    21: 'Soybean___healthy', 22: 'Strawberry___Leaf_scorch',
    23: 'Strawberry___healthy', 24: 'Tomato___Bacterial_spot',
    25: 'Tomato___Early_blight', 26: 'Tomato___Late_blight',
    27: 'Tomato___Leaf_Mold', 28: 'Tomato___Septoria_leaf_spot',
    29: 'Tomato___Spider_mites_Two-spotted_spider_mite', 30: 'Tomato___Target_Spot',
    31: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 32: 'Tomato___Tomato_mosaic_virus',
    33: 'Tomato___healthy'
}

PLANT_TO_DISEASE_PREFIX = {
    'Apple': 'Apple',
    'Cherry': 'Cherry_(including_sour)',
    'Corn': 'Corn_(maize)',
    'Grape': 'Grape',
    'Orange': 'Orange',
    'Peach': 'Peach',
    'Pepper_bell': 'Pepper,_bell',
    'Potato': 'Potato',
    'Soybean': 'Soybean',
    'Strawberry': 'Strawberry',
    'Tomato': 'Tomato'
}
WEATHER_API_KEY = "1020a5b033aee42c4874144d88e5dade"
GEMINI_API_KEY = "AIzaSyBqQDeTQ_RwTvrsjz8D9XtozGUWw2vZoIk"
OPENCAGE_API_KEY = 'cb0e84b387ca439e973f121ae101cecc'

leaf_detector = LeafDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_plant(img_path, target_size):
    img = image.load_img(
        img_path, 
        color_mode='rgb',
        target_size=target_size,
        interpolation='bilinear'
    )
    img_array = image.img_to_array(img)  # Values 0-255
    return np.expand_dims(img_array, axis=0)

def get_weather_data(latitude, longitude):
    try:
        response = requests.get(
            f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}&units=metric',
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return [
                data['main']['temp'],
                data['main']['humidity'],
                data.get('rain', {}).get('1h', 0)
            ]
    except Exception as e:
        print(f"Weather API error: {str(e)}")
    return [25.0, 60.0, 0.0]


def preprocess_image_for_disease(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def predict_disease(plant_type, img_path, weather_data):
    """Predict plant disease with weather context"""
    try:
        interpreter = load_model("disease")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Debug input details
        print("\n=== Disease Model Input Details ===")
        print(f"Input 0: {input_details[0]['name']} - Shape: {input_details[0]['shape']}")
        print(f"Input 1: {input_details[1]['name']} - Shape: {input_details[1]['shape']}")

        # Preprocess image (shape: [1, 224, 224, 3])
        img_array = preprocess_image_for_disease(img_path, (224, 224))
        print(f"Image array shape: {img_array.shape}")

        # Prepare weather data (shape: [1, 3])
        weather_array = np.array(weather_data, dtype=np.float32).reshape(1, 3)
        print(f"Weather array shape: {weather_array.shape}")

        # Verify input shapes match model expectations
        if img_array.shape != tuple(input_details[1]['shape']):
            raise ValueError(f"Image shape mismatch. Model expects {input_details[1]['shape']}, got {img_array.shape}")

        if weather_array.shape != tuple(input_details[0]['shape']):
            raise ValueError(f"Weather shape mismatch. Model expects {input_details[0]['shape']}, got {weather_array.shape}")

        # Set tensors with corrected order
        interpreter.set_tensor(input_details[1]['index'], img_array)  # Image to second input
        interpreter.set_tensor(input_details[0]['index'], weather_array)  # Weather to first input
        
        interpreter.invoke()

        # --------------------------------------------------
        # 5. Process Output (Original Logic)
        # --------------------------------------------------
        logits = interpreter.get_tensor(output_details[0]['index'])[0]
        
        disease_prefix = PLANT_TO_DISEASE_PREFIX.get(plant_type, '')
        valid_indices = [idx for idx, label in DISEASE_LABELS.items() 
                        if label.startswith(f"{disease_prefix}___")]
        
        if not valid_indices:
            return [('Unknown Disease', 1.0)], "high"

        valid_logits = logits[valid_indices]
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        probs = exp_logits / exp_logits.sum()
        
        predictions = sorted(
            [(DISEASE_LABELS[valid_indices[i]], float(probs[i])) 
            for i in range(len(valid_indices))],
            key=lambda x: -x[1]
        )

        top_confidence = predictions[0][1] if predictions else 0
        confidence_level = "high" if top_confidence > 0.65 else \
                         "medium" if (top_confidence - predictions[1][1] > 0.15) else "low"
        
        clear_memory()
        return predictions[:TOP_K], confidence_level

    except Exception as e:
        clear_memory()
        print(f"Disease prediction error: {str(e)}")
        return [], "unknown"

def get_gemini_recommendation(disease_name, weather_data):
    if "healthy" in disease_name.lower():
        return "Plant is healthy. Maintain current care practices."
    
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": f"Provide treatment for {disease_name} considering: {weather_data[0]}°C temp, {weather_data[1]}% humidity"}]}]},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text).replace('•', '➜')
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return "Recommendation unavailable. Please consult an agricultural expert."

def reverse_geocoding(latitude, longitude):
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.reverse_geocode(latitude, longitude)
        if results:
            components = results[0]['components']
            return ", ".join(filter(None, [
                components.get("road"),
                components.get("city"),
                components.get("state"),
                components.get("country")
            ]))
        return "Location unavailable"
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return "Service unavailable"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        if not leaf_detector.detect(save_path):
            return jsonify({"status": "error", "message": "Please upload a clear plant leaf image"}), 400

        interpreter = load_model("plant")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        img_array = preprocess_image_for_plant(save_path, (128, 128))
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        top_k_idx = np.argsort(predictions)[-TOP_K:][::-1]
        plant_preds = [(PLANT_CLASS_NAMES[i], float(predictions[i])) for i in top_k_idx]
        best_plant, best_conf = plant_preds[0]

        response_data = {
            "predictions": [{"class": p[0], "confidence": p[1]} for p in plant_preds],
            "top_confidence": best_conf,
            "filename": filename,
            "status": "direct_success" if best_conf >= HIGH_CONFIDENCE_THRESHOLD else "needs_confirmation",
            "final_prediction": best_plant if best_conf >= HIGH_CONFIDENCE_THRESHOLD else None,
            "message": "High confidence prediction" if best_conf >= HIGH_CONFIDENCE_THRESHOLD 
                      else "Please confirm plant type"
        }

        clear_memory()
        return jsonify(response_data)

    except Exception as e:
        clear_memory()
        return jsonify({"status": "error", "message": f"Analysis failed: {str(e)}"}), 500

@app.route('/confirm_plant', methods=['POST'])
def confirm_plant():
    try:
        data = request.get_json()
        selected_plant = data.get('plant')
        filename = data.get('filename')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not selected_plant or selected_plant == "not_listed":
            return jsonify({"status": "error", "message": "Invalid plant selection"}), 400
        
        weather_data = get_weather_data(latitude, longitude)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        disease_preds, confidence_level = predict_disease(selected_plant, save_path, weather_data)

        results = [{
            "name": name,
            "confidence": confidence,
            "recommendation": get_gemini_recommendation(name, weather_data)
        } for name, confidence in disease_preds]

        return jsonify({
            "status": "success",
            "plant": selected_plant,
            "diseases": results,
            "warnings": ["Low confidence results - consider expert consultation"] if confidence_level == "low" else [],
            "location": reverse_geocoding(latitude, longitude)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
