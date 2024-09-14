import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = 128

def load_images(data_dir):
    images = []
    labels = []
    for label in ['Ripe', 'Unripe']:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue
        class_num = 0 if label == 'Ripe' else 1
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0  # Ensure the image is in float32 and normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
            labels.append(class_num)
    return np.array(images), np.array(labels)

def plot_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title("Ripe" if labels[i] == 0 else "Unripe")
        plt.axis('off')
    plt.show()

data_dir = './tomato_dataset'
images, labels = load_images(data_dir)
plot_images(images, labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

# Continue with further steps, like building and training the model
