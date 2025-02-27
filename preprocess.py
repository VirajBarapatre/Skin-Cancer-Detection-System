import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Metadata
print("Loading Metadata...")
metadata_path = "dataset/metadata.csv"  # Metadata CSV File Path
df = pd.read_csv(metadata_path)
print("Metadata Loaded Successfully:")
print(df.head())

# Image Folder Path
image_dir = "dataset/SkinCancer"  # Images Folder Path

# Image Size
image_size = (128, 128)  # Resize all images to 128x128

# Initialize Lists
images = []
labels = []

# Loop through each row in the CSV file
print("\nProcessing Images...")

for index, row in df.iterrows():
    image_name = row["image_id"] + ".jpg"
    image_path = os.path.join(image_dir, image_name)

    if os.path.exists(image_path):
        # Read Image
        img = cv2.imread(image_path)
        img = cv2.resize(img, image_size)  # Resize
        img = img / 255.0  # Normalize between 0 and 1
        
        # Append Image and Label
        images.append(img)
        labels.append(row["dx"])  # 'dx' is the label column

    else:
        print(f"Image not found: {image_path}")

# Convert Lists to NumPy Arrays
images = np.array(images)
labels = np.array(labels)

print(f"\nTotal Images Loaded: {len(images)}")
print(f"Total Labels Loaded: {len(labels)}")

# Encode Labels
print("\nEncoding Labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Save Label Classes
np.save("dataset/label_classes.npy", label_encoder.classes_)
print("Labels Encoded Successfully")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save Arrays
np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)
# Save label encoder correctly
np.save('dataset/label_classes.npy', label_encoder.classes_)


print("\n✅ Dataset Preprocessing Completed Successfully!")
print(f"Training Images: {len(X_train)}")
print(f"Testing Images: {len(X_test)}")
