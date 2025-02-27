import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
X_train = np.load("dataset/X_train.npy")
X_test = np.load("dataset/X_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")
label_classes = np.load("dataset/label_classes.npy", allow_pickle=True)

# Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(label_classes), activation='softmax')  # Multi-class classification
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("model.h5")
print("Model training completed and saved as 'model.h5'")
