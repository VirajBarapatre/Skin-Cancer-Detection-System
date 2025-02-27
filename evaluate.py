import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")
label_classes = np.load("dataset/label_classes.npy", allow_pickle=True)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print Evaluation Results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_classes))
