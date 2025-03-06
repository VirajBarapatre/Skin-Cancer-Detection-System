Skin Cancer Detection System
A Deep Learning-based Skin Cancer Detection System using Convolutional Neural Networks (CNN) to classify skin cancer images into seven categories with a GUI interface.

---

📌 Project Overview
This project uses TensorFlow and Keras to classify skin cancer images based on the HAM10000 Dataset. It provides a Graphical User Interface (GUI) built with Tkinter to allow users to upload images and get real-time predictions.

---

🔑 Key Features
Image Preprocessing using OpenCV and NumPy
CNN Model for Image Classification
Model Training and Evaluation
GUI Interface for User Interaction
Visualization of Training Accuracy & Loss

---

🗂 Folder Structure

Skin_Cancer_Detection_System/
├── dataset/                  # Dataset Folder
│   ├── images.npy            # Preprocessed Images
│   ├── labels.npy            # Preprocessed Labels
│   ├── X_test.npy            # Test Images
│   └── y_test.npy            # Test Labels
│
├── models/                   # Trained Model Folder
│   └── skin_cancer_model.h5   # Saved Model
│
├── utils/                    # Utility Functions
│   └── preprocess.py         # Preprocessing Code
│
├── gui.py                    # GUI Interface
├── train_model.py            # Model Training Code
├── evaluate.py               # Model Evaluation Code
├── requirements.txt          # Required Libraries
├── README.md                 # Project Documentation
└── LICENSE                   # License File
---

📊 Dataset
Dataset Name: HAM10000

Classes:
Melanoma
Melanocytic Nevi
Basal Cell Carcinoma
Actinic Keratoses
Benign Keratosis
Dermatofibroma
Vascular Lesions

---

🔌 Installation

Step 1: Clone Repository
git clone https://github.com/VirajBarapatre/Skin-Cancer-Detection-System.git
cd Skin-Cancer-Detection-System

Step 2: Install Dependencies
pip install -r requirements.txt

---

🏃‍♀ How to Run

1. Preprocess the Dataset:
python utils/preprocess.py

2. Train the Model:
python train_model.py

4. Evaluate the Model:
python evaluate.py

5. Run the GUI:
python gui.py

---

📌 Results
Model Accuracy: 90%
Confusion Matrix
Training & Validation Accuracy Graph

---

🛠 Tech Stack

Python
TensorFlow
Keras
NumPy
OpenCV
Tkinter
Matplotlib

---

📄 License
This project is licensed under the MIT License.

---

🌐 Connect with Me
GitHub: VirajBarapatre
