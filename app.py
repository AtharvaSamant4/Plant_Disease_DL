import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from PIL import ImageOps
import requests
import re
from opencage.geocoder import OpenCageGeocode

# Configure TensorFlow for low memory usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Model paths (using TFLite models from repo)
MODEL_PATHS = {
    "leaf": "Plant_vs_Nonplant.tflite",
    "plant": "Plant_Classification_Model.tflite",
    "disease": "Plant_Disease_Predictor_with_Weather.tflite"
}

# Lazy-loaded models
_MODELS = {
    "leaf": None,
    "plant": None,
    "disease": None
}

def load_model(model_name):
    """Lazy-load TFLite models"""
    if _MODELS[model_name] is None:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS[model_name])
        interpreter.allocate_tensors()
        _MODELS[model_name] = interpreter
    return _MODELS[model_name]

def clear_memory():
    """Release model resources after each prediction"""
    for key in _MODELS:
        _MODELS[key] = None
    tf.keras.backend.clear_session()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TOP_K = 3
HIGH_CONFIDENCE_THRESHOLD = 0.9
PLANT_CLASS_NAMES = ['Apple', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 
                    'Pepper_bell', 'Potato', 'Soybean', 'Strawberry', 'Tomato']

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

DISEASE_LABELS = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Cherry_(including_sour)___Powdery_mildew',
    5: 'Cherry_(including_sour)___healthy',
    6: 'Corn_(maize)___Common_rust_',
    7: 'Corn_(maize)___Northern_Leaf_Blight',
    8: 'Corn_(maize)___healthy',
    9: 'Grape___Black_rot',
    10: 'Grape___Esca_(Black_Measles)',
    11: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    12: 'Grape___healthy',
    13: 'Orange___Haunglongbing_(Citrus_greening)',
    14: 'Peach___Bacterial_spot',
    15: 'Peach___healthy',
    16: 'Pepper,_bell___Bacterial_spot',
    17: 'Pepper,_bell___healthy',
    18: 'Potato___Early_blight',
    19: 'Potato___Late_blight',
    20: 'Potato___healthy',
    21: 'Soybean___healthy',
    22: 'Strawberry___Leaf_scorch',
    23: 'Strawberry___healthy',
    24: 'Tomato___Bacterial_spot',
    25: 'Tomato___Early_blight',
    26: 'Tomato___Late_blight',
    27: 'Tomato___Leaf_Mold',
    28: 'Tomato___Septoria_leaf_spot',
    29: 'Tomato___Spider_mites_Two-spotted_spider_mite',
    30: 'Tomato___Target_Spot',
    31: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    32: 'Tomato___Tomato_mosaic_virus',
    33: 'Tomato___healthy'
}

# API Keys (Consider using environment variables in production)
WEATHER_API_KEY = "1020a5b033aee42c4874144d88e5dade"
GEMINI_API_KEY = "AIzaSyBqQDeTQ_RwTvrsjz8D9XtozGUWw2vZoIk"
OPENCAGE_API_KEY = 'cb0e84b387ca439e973f121ae101cecc'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_plant(img_path, target_size):
    """Preprocess image for plant classification"""
    img = image.load_img(img_path, target_size=target_size)
    img_gray = ImageOps.grayscale(img)
    img_array = np.array(img_gray)
    return np.expand_dims(np.stack([img_array] * 3, axis=-1), axis=0)

def preprocess_image_for_disease(img_path, target_size):
    """Preprocess image for disease prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def get_weather_data(latitude, longitude):
    """Fetch weather data from OpenWeatherMap"""
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}&units=metric'
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [
                data['main']['temp'],
                data['main']['humidity'],
                data.get('rain', {}).get('1h', 0)
            ]
    except Exception as e:
        print(f"Weather API error: {str(e)}")
    return [25.0, 60.0, 0.0]  # Default values

def detect_leaf(image_path):
    """Detect if image contains a plant leaf"""
    interpreter = load_model("leaf")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img_array = preprocess_image_for_plant(image_path, (224, 224))
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    clear_memory()
    return prediction[0][0]

def predict_disease(plant_type, img_path, weather_data):
    """Predict plant disease with weather context"""
    try:
        interpreter = load_model("disease")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preprocess inputs
        img_array = preprocess_image_for_disease(img_path, (224, 224))
        weather_array = np.array(weather_data).reshape(1, -1).astype(np.float32)
        
        # Set multi-input tensors
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.set_tensor(input_details[1]['index'], weather_array)
        interpreter.invoke()
        
        logits = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Filter valid diseases for plant type
        disease_prefix = PLANT_TO_DISEASE_PREFIX.get(plant_type, '')
        valid_indices = [idx for idx, label in DISEASE_LABELS.items() 
                        if label.startswith(f"{disease_prefix}___")]
        
        if not valid_indices:
            return [('Unknown Disease', 1.0)], "high"
            
        # Process predictions
        valid_logits = logits[valid_indices]
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        probs = exp_logits / exp_logits.sum()
        
        predictions = sorted(
            [(DISEASE_LABELS[valid_indices[i]], float(probs[i])) 
            for i in range(len(valid_indices))],
            key=lambda x: -x[1]
        )

        # Confidence analysis
        top_confidence = predictions[0][1] if predictions else 0
        confidence_gap = top_confidence - predictions[1][1] if len(predictions) > 1 else 0
        
        confidence_level = "high" if top_confidence > 0.65 else \
                         "medium" if confidence_gap > 0.15 else "low"
        
        clear_memory()
        return predictions[:TOP_K], confidence_level

    except Exception as e:
        clear_memory()
        print(f"Disease prediction error: {str(e)}")
        return [], "unknown"

def get_gemini_recommendation(disease_name, weather_data):
    """Get treatment recommendations from Gemini API"""
    if "healthy" in disease_name.lower():
        return "Plant is healthy. Maintain current care practices."
    
    temp, humidity, rainfall = weather_data
    prompt = (f"Provide treatment for {disease_name} considering: "
              f"{temp}°C temp, {humidity}% humidity, {rainfall}mm rain. "
              "Give 4 concise bullet points without markdown.")
    
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json= {"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return re.sub(r'\*\*(.*?)\*\*', r'\1', text).replace('•', '➜')
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return "Recommendation unavailable. Please consult an agricultural expert."

def reverse_geocoding(latitude, longitude):
    """Convert coordinates to readable location"""
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.reverse_geocode(latitude, longitude)
        return ", ".join([str(results[0]['components'].get(c, '') )
                        for c in ["road", "city", "state", "country"]]) if results else "Location unavailable"
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return "Service unavailable"

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def about():
    return render_template('results.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')

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

        # Leaf detection
        if detect_leaf(save_path) > 0.5:
            return jsonify({"status": "error", "message": "Please upload a clear plant leaf image"}), 400

        # Plant classification
        interpreter = load_model("plant")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()  # ADD THIS LINE
        
        img_array = preprocess_image_for_plant(save_path, (128, 128))
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        
        # CORRECTED PREDICTION LINE
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
    app.run(host='0.0.0.0', port=5000)
