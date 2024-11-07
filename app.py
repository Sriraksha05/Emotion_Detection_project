from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import joblib
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model and scaler
model = joblib.load('emotion_model.joblib')
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')


def extract_features(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def predict_emotion(file_path):
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    emotion_pred = model.predict(features_scaled)
    return le.inverse_transform(emotion_pred)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if audio_file:
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        try:
            emotion = predict_emotion(filepath)
            os.remove(filepath)  # Remove the file after prediction
            return jsonify({'emotion': emotion})
        except Exception as e:
            os.remove(filepath)  # Ensure file is removed even if an error occurs
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)