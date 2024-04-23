import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to load and preprocess image data

def load_data(data_dir):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            
            # Attempt to load the image
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to load image '{image_path}'")
                    continue  # Skip to the next image if loading fails
            except Exception as e:
                print(f"Error: {e}")
                continue  # Skip to the next image if an exception occurs
            
            # Attempt to resize the image
            try:
                image = cv2.resize(image, (100, 100)) # Resize image to (100, 100)
            except Exception as e:
                print(f"Error: Unable to resize image '{image_path}': {e}")
                continue  # Skip to the next image if resizing fails
            
            images.append(image)
            labels.append(label)
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    
    return images, labels


# Load and preprocess data
data_dir = r"\dataset"  # Directory containing your dataset
images, labels = load_data(data_dir)

# Convert labels to one-hot encoding
label_dict = {"with_mask": 0, "without_mask": 1}
labels = np.array([label_dict[label] for label in labels])
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('mask_detection_model.h5')
