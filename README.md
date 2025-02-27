# Skin Cancer Detection System
### Automated Skin Cancer Detection using Deep Learning

This project is an **AI-based Skin Cancer Detection System** that classifies skin lesion images into seven different types using **Deep Learning (TensorFlow)** and provides an easy-to-use **GUI interface with Tkinter**.

---

## 🧠 Technologies Used
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Tkinter (GUI)
- Matplotlib
- PIL (Pillow)

---

## 📌 Dataset
The dataset used is the **HAM10000 Skin Lesions Dataset** from Kaggle.

Link: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---
📄 Project File Tree
Skin-Cancer-Detection-System/
│
├── dataset/                  # Dataset Folder
│   ├── images.npy            # Preprocessed Image Data
│   ├── labels.npy            # Preprocessed Labels
│   ├── X_test.npy            # Test Images
│   ├── X_train.npy           # Training Images
│   ├── y_test.npy            # Test Labels
│   ├── y_train.npy           # Training Labels
│   └── label_classes.npy      # Encoded Label Classes
│
├── model/                    # Saved Model Folder
│   └── skin_cancer_model.h5   # Trained Model
│
├── gui.py                    # Tkinter GUI Code
├── preprocess.py             # Data Preprocessing Code
├── train_model.py            # Model Training Code
├── evaluate.py               # Model Evaluation Code
├── README.md                 # Project Documentation
├── requirements.txt          # Required Libraries
├── LICENSE                   # License File
└── .gitignore                # Git Ignore File

---

## 🔑 Features
- Skin Cancer Image Classification into 7 categories:
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)
  
- Model Training and Evaluation
- Real-Time Image Prediction with GUI
- Accuracy and Loss Visualization
- Save & Load Trained Model Automatically

---

## ⚙️ Installation
### Clone Repository
```bash
git clone https://github.com/VirajBarapatre/Skin-Cancer-Detection-System.git
cd Skin-Cancer-Detection-System

Install Dependencies
pip install -r requirements.txt


🛠️ How to Run Project?

Preprocess Dataset
python preprocess.py

Train Model
python train_model.py

Evaluate Model
python evaluate.py


GUI for Skin Cancer Detection
python gui.py

📊 Model Performance
Metric	Value
Accuracy	75.2%
Loss	0.89
Precision	76.1% 
Recall	74.3%

🖼️ GUI Preview
Skin Cancer Detection GUI

📄 License
This project is licensed under the MIT License.

✍️ Author
Viraj Barapatre
LinkedIn: Viraj Barapatre
GitHub: VirajBarapatre
