# 🌿 Plant Disease Detection with Weather-Aware Deep Learning (TFLite Version)

This project is a lightweight yet powerful web app for plant disease detection using deep learning models converted to TensorFlow Lite. It takes a plant leaf image as input and enhances the disease prediction by factoring in real-time weather data from the user’s location. The app returns Top-K disease predictions along with actionable recommendations.

## 🚀 Key Features

- ✅ **Leaf Verification**: Checks if the uploaded image is a plant leaf.
- 🌱 **Plant Type Classification**: Identifies the plant (apple, grape, corn, potato, etc).
- 🧠 **Unified Disease Prediction**: A single TFLite model  predicts the disease with weather data as additional input
- ☁️ **Live Weather Integration**: Automatically fetches temperature, humidity, etc., based on user location
- 🎯 **Top-K Predictions**: Displays multiple possible diseases with confidence scores
- 🌐 **Flask Web App Interface**: User-friendly UI for image upload and results


## 🔧 Installation & Setup

1. Clone the repository
    --git clone https://github.com/yourusername/plant-disease-weather-tflite.git

2. Install the required packages
   -- pip install -r requirements.txt

3. Start the Flask server
   -- python app.py


🛠 Tech Stack
Python, Flask
TensorFlow Lite
HTML, CSS
OpenWeatherMap API

📌 To-Do / Enhancements
Add UI for mobile
Expand to more plants
Upload history for users


👤 Author
Atharva Samant
https://github.com/AtharvaSamant4


