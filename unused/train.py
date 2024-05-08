import cv2
import os
import numpy as np

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))  # Resize image to a fixed size
    return resized

# Function to extract features (LBPH)
def extract_features(images):
    faces = []
    labels = []
    label_id = 0
    label_map = {}

    for root, dirs, files in os.walk(images):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith('.JPG'):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(f"Processing image: {path} {label}")
                  # Get label from folder name
                if label not in label_map:
                    label_map[label] = label_id
                    label_id += 1
                id_ = label_map[label]

                # Preprocess image
                face = preprocess_image(path)

                # Add face and label to lists
                faces.append(face)
                labels.append((id_, label))  # Store both numerical ID and text label

    return faces, labels

# Train LBPH face recognizer
def train_face_recognizer(faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array([label[0] for label in labels]))  # Use only numerical IDs for training
    recognizer.save("trained_model.yml")
    print("Model trained and saved successfully.")

    # Print labels text after training
    print("Labels text:")
    for label_id, label_text in labels:
        print(f"Label ID: {label_id}, Label Text: {label_text}")

# Example usage
images_folder = "dataset"
faces, labels = extract_features(images_folder)
train_face_recognizer(faces, labels)
