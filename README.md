# 🥕 Rot Check: AI-Driven Quality Assurance

**Rot Check** is an end-to-end computer vision pipeline designed to automate **carrot sorting for packaging**.  
By combining a **Convolutional Neural Network (CNN)** with **traditional computer vision metrics**, the system delivers robust **Healthy vs. Spoiled** classification along with detailed **quality grading**.

---

## 📊 Performance & Data Strategy

To ensure **production-level reliability**, the model was trained on the **Food Freshness Dataset** -  https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset with the following optimizations:

- **Training Accuracy**: 85.2%  
- **Validation Accuracy (Peak)**: 87.2%  
- **Class Imbalance Handling**: Applied a **1:4 class weight** to prioritize detection of spoiled carrots  
- **Data Augmentation**: Rotation, zoom, and flips to improve generalization  
- **Validation Strategy**: 20% validation split to test performance on unseen data  

---

## 🛠️ Key Features

### 🤖 Deep Learning Classifier
- Binary CNN model built using **TensorFlow / Keras**
- Classifies carrots as **Healthy** or **Spoiled**

### 🧪 Hybrid Quality Scoring
In addition to CNN predictions, the system computes:
- **Color Score** – Euclidean distance-based color deviation  
- **Shape Score** – Aspect ratio analysis  
- **Size Score** – Relative size estimation  

### 🔁 Active Learning Loop
- Real-time **user feedback** after prediction  
- Misclassified images are **stored for future retraining**
- Improves model performance over time

### 🌐 Web Dashboard
- Intuitive **Flask-based interface**
- Instant classification and quality grading results

---

## 📸 Demo

### Landing Page
![Landing Page](assets/Landing%20page.png)

### Healthy Classification
![Healthy Classification](assets/Healthy%20classification.png)

### Rotten Classification
![Rotten Classification](assets/Rotten%20classification.png)

---

## 🧪 Technologies Used

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask  
- **Machine Learning**: Python, TensorFlow (CNN Model)  

---

## 🚀 Quick Start

### 🔧 Installation
```bash
pip install -r requirements.txt
```

### 🧠 Train the Model
```bash
python train_model.py
```

Trained model is saved as: carrot_classifier_model.h5

### ▶️ Run the Application
```bash
python app.py
```

## 📖 How to Use

1. Open your browser and navigate to: http://127.0.0.1:5000
2. Upload an image of a carrot.
3. The model classifies whether the carrot is suitable for packaging.
4. Confirm whether the prediction was **correct** or **incorrect**.
5. The system learns from your feedback and improves over time.

---

## 🌱 Future Enhancements

- Multi-class grading (A / B / C quality)
- Batch image processing
- Edge deployment for real-time factory usage
- Automated retraining pipelines

---

## 📌 Project Goal

To reduce manual inspection errors, increase sorting efficiency, and enable AI-driven quality assurance in agricultural packaging workflows.

