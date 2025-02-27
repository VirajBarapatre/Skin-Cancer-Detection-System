import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('model.h5')
label_classes = np.load('dataset/label_classes.npy', allow_pickle=True)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return label_classes[predicted_class]

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}", font=("Arial", 14), fg="green")

# GUI Setup
root = tk.Tk()
root.title("Skin Cancer Detection System")
root.geometry("500x600")
root.config(bg="#f4f4f4")

title_label = Label(root, text="Skin Cancer Detection System", font=("Arial", 20), bg="#f4f4f4", pady=20)
title_label.pack()

upload_btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#0078D7", fg="white")
upload_btn.pack(pady=10)

panel = Label(root)
panel.pack(pady=20)

result_label = Label(root, text="", bg="#f4f4f4")
result_label.pack(pady=20)

root.mainloop()
