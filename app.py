import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

# Ensure upload directories exist
os.makedirs('uploads/healthy', exist_ok=True)
os.makedirs('uploads/spoiled', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Load the pre-trained model once when app starts
model = tf.keras.models.load_model('carrot_classifier_model.h5')

# Function to classify carrot image
def classify_carrot(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0]

# Function to calculate quality scores
def calculate_quality_score(img):
    img = img.resize((150, 150))
    img_array = np.array(img)
    avg_color = np.mean(img_array, axis=(0, 1))
    color_score = np.linalg.norm(avg_color - np.array([255, 102, 0]))
    aspect_ratio = img_array.shape[1] / img_array.shape[0]
    shape_score = 1 - abs(aspect_ratio - 1.5)
    size_score = img_array.size / (150 * 150 * 3)
    total_score = (1/color_score) + (1/shape_score) + (1/size_score)
    return color_score, shape_score, size_score, total_score

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_carrot', methods=['POST'])
def classify():
    if 'carrotImage' not in request.files:
        return "No file uploaded.", 400

    file = request.files['carrotImage']
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = Image.open(img_path)
    prediction = classify_carrot(img)
    classification = "healthy" if prediction < 0.5 else "spoiled"

    color_score, shape_score, size_score, total_score = calculate_quality_score(img)

    if classification == "healthy":
        actual_img_path = os.path.join('uploads', file.filename)
    else:
        os.rename(img_path, os.path.join('uploads/spoiled', file.filename))
        actual_img_path = os.path.join('uploads/spoiled', file.filename)

    return render_template('result.html', classification=classification, img_path=actual_img_path,
                           color_score=color_score, shape_score=shape_score, 
                           size_score=size_score, total_score=total_score)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/feedback', methods=['POST'])
def feedback(): 
    img_path = request.form['img_path']
    feedback = request.form['feedback']

    if feedback == "correct":
        return "Thank you for your feedback! The image has been recorded."

    if feedback == "healthy":
        new_path = os.path.join('uploads/healthy', os.path.basename(img_path))
    else:
        new_path = os.path.join('uploads/spoiled', os.path.basename(img_path))

    os.rename(img_path, new_path)

    return "Thank you for your feedback! The image has been recorded."

# Run the Flask app
if __name__ == '__main__':
    # Host set to 127.0.0.1 and debug disabled for Electron
    app.run(host="127.0.0.1", port=5000, debug=False)