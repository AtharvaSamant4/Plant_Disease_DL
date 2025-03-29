import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from PIL import ImageOps
import requests
import json
import re
from opencage.geocoder import OpenCageGeocode

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

# Load models
LEAF_MODEL = tf.keras.models.load_model("plant_vs_non_plant_model.h5")
PLANT_MODEL = tf.keras.models.load_model("Plant_Classification_Architecture(Apple,Banana,etc).keras")
DISEASE_MODEL = tf.keras.models.load_model("Plant_Disease_Predictor_with_Weather.keras")

# API Keys
WEATHER_API_KEY = "1020a5b033aee42c4874144d88e5dade"
GEMINI_API_KEY = "AIzaSyBqQDeTQ_RwTvrsjz8D9XtozGUWw2vZoIk"
OPENCAGE_API_KEY = 'cb0e84b387ca439e973f121ae101cecc'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_plant(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_gray = ImageOps.grayscale(img)
    img_array = np.array(img_gray)
    img_array = np.stack([img_array] * 3, axis=-1)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_for_disease(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def get_predictions(model, img_array):
    predictions = model.predict(img_array)[0]
    top_k_idx = np.argsort(predictions)[-TOP_K:][::-1]
    return [(PLANT_CLASS_NAMES[i], float(predictions[i])) for i in top_k_idx]

def get_weather_data(latitude, longitude):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}&units=metric'
    try:
        response = requests.get(url)
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
    img_array = preprocess_image_for_plant(image_path, (224, 224))
    return LEAF_MODEL.predict(img_array)[0][0]

def predict_disease(plant_type, img_path, weather_data, temperature=0.3):
    try:
        # Preprocess inputs
        img_array = preprocess_image_for_disease(img_path, (224, 224))
        weather_array = np.array(weather_data).reshape(1, -1)
        
        # Get predictions
        logits = DISEASE_MODEL.predict([img_array, weather_array])[0]
        
        # Apply temperature scaling with tie-breaking
        scaled_logits = logits / temperature
        scaled_logits += np.random.normal(0, 1e-6, scaled_logits.shape)  # Break exact ties
        
        # Filter valid diseases for the plant type
        disease_prefix = PLANT_TO_DISEASE_PREFIX.get(plant_type, '')
        valid_indices = [idx for idx, label in DISEASE_LABELS.items() 
                        if label.startswith(f"{disease_prefix}___")]
        
        if not valid_indices:
            return [('Unknown Disease', 1.0)], "high"
            
        # Apply softmax to valid classes
        valid_logits = scaled_logits[valid_indices]
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        probs = exp_logits / exp_logits.sum()
        
        # Create sorted predictions
        predictions = sorted(
            [(DISEASE_LABELS[valid_indices[i]], float(probs[i])) 
            for i in range(len(valid_indices))],
            key=lambda x: -x[1]
        )

        # Confidence analysis
        top_confidence = predictions[0][1] if predictions else 0
        confidence_gap = top_confidence - predictions[1][1] if len(predictions) > 1 else 0
        
        # Determine confidence level
        if top_confidence > 0.65:
            confidence_level = "high"
        elif confidence_gap > 0.15:
            confidence_level = "medium"
        else:
            confidence_level = "low"
            predictions = predictions[:3]  # Return only top prediction when uncertain

        return predictions[:TOP_K], confidence_level

    except Exception as e:
        print(f"Disease prediction error: {str(e)}")
        return [], "unknown"

def clean_markdown(text):
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def get_gemini_recommendation(disease_name, weather_data):
    if "healthy" in disease_name.lower():
        return "Plant is healthy. Maintain current care practices."
    
    temp, humidity, rainfall = weather_data
    prompt = (f"Provide treatment for {disease_name} considering: "
              f"{temp}°C temp, {humidity}% humidity, {rainfall}mm rain. "
              "Give 4 concise bullet points without markdown.")
    
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return clean_markdown(text).replace('•', '➜')
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return "Recommendation unavailable. Please consult an agricultural expert."

def reverse_geocoding(latitude, longitude):
    try:
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        results = geocoder.reverse_geocode(latitude, longitude)
        
        if results:
            return normalize_address(results[0]['components'])
        
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

@app.route('/results.html')
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

        # Leaf detection
        leaf_value = detect_leaf(save_path)
        if leaf_value > 0.5:
            return jsonify({"status": "error", "message": "Please upload a clear plant leaf image"}), 400

        # Plant classification
        img_array = preprocess_image_for_plant(save_path, (128, 128))
        plant_preds = get_predictions(PLANT_MODEL, img_array)
        best_plant, best_conf = plant_preds[0]

        # Print plant predictions
        print("\n=== Top Plant Predictions ===")
        for i, (plant, conf) in enumerate(plant_preds, 1):
            print(f"{i}. {plant}: {conf*100:.2f}%")

        response_data = {
            "predictions": [{"class": p[0], "confidence": p[1]} for p in plant_preds],
            "top_confidence": best_conf,
            "filename": filename
        }

        if best_conf >= HIGH_CONFIDENCE_THRESHOLD:
            response_data.update({
                "status": "direct_success",
                "final_prediction": best_plant,
                "message": "High confidence prediction - proceeding to disease detection"
            })
        else:
            response_data.update({
                "status": "needs_confirmation",
                "message": "Please confirm plant type"
            })

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"status": "error", "message": f"Analysis failed: {str(e)}"}), 500

@app.route('/confirm_plant', methods=['POST'])
def confirm_plant():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Invalid request format"}), 400
            
        data = request.get_json()
        selected_plant = data.get('plant')
        filename = data.get('filename')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not selected_plant:
            return jsonify({"status": "error", "message": "No plant selected"}), 400
        
        if selected_plant == "not_listed":
            return jsonify({"status": "unknown_plant", "message": "This plant is not in our database"}), 400
        
        # Get weather data
        weather_data = get_weather_data(latitude, longitude)
        
        # Get disease predictions
        # [28.5, 85.0, 5.0] for Corn Common Rust
        # [25.0, 90.0, 6.0] for potato early blight
        # [17.0, 92.0, 8.0] for apple scab
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        disease_preds, confidence_level = predict_disease(selected_plant, save_path,[25.0, 90.0, 6.0])

        # Print disease predictions
        print(f"\n=== Disease Predictions ({confidence_level.capitalize()} Confidence) ===")
        for i, (name, conf) in enumerate(disease_preds, 1):
            print(f"{i}. {name}: {conf*100:.2f}%")

        # Generate recommendations
        results = []
        warnings = []
        for name, confidence in disease_preds:
            results.append({
                "name": name,
                "confidence": confidence,
                "recommendation": get_gemini_recommendation(name, weather_data)
            })
        
        if confidence_level == "low":
            warnings.append("Low confidence results - consider expert consultation")

        return jsonify({
            "status": "success",
            "plant": selected_plant,
            "diseases": results,
            "warnings": warnings,
            "location": reverse_geocoding(latitude, longitude),
            "message": "Plant confirmed - Disease detection completed"
        })
       
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def normalize_address(raw_address):
    # Standardize administrative hierarchy
    components = [
        "road", "suburb", "city_district", "city", 
        "state", "postcode", "country"
    ]
    
    standardized = []
    for comp in components:
        if comp in raw_address:
            standardized.append(str(raw_address[comp]))
    
    return ", ".join(standardized)


# @app.route('/disease_results')
# def disease_results():
#     try:
#         plant = request.args.get('plant', 'Unknown Plant')
#         diseases_json = request.args.get('diseases', '[]')
#         diseases = json.loads(diseases_json)
#         return render_template('disease.html', plant=plant, diseases=diseases)
#     except Exception as e:
#         return render_template('error.html', message="Could not load results")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)