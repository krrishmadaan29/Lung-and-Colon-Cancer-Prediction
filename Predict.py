import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('alexnet_lung_colon_final.h5')

# Class names (should match your training classes)
CLASSES = [
    'colon_aca',    # Colon adenocarcinoma
    'colon_n',      # Benign colon tissue
    'lung_aca',     # Lung adenocarcinoma
    'lung_n',       # Benign lung tissue
    'lung_scc'      # Lung squamous cell carcinoma
]

# Image preprocessing functions (must match training)
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to AlexNet input size
    img = cv2.resize(img, (227, 227))
    
    # Apply enhancement
    img = enhance_image(img)
    
    # Scale to [0,1] and normalize
    img = img / 255.0
    img = (img - np.mean(img)) / np.std(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image_class(image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class = CLASSES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.show()
    
    # Print results
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nClass probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"{CLASSES[i]}: {prob*100:.2f}%")
    
    return predicted_class, confidence

# Example usage - ADD YOUR IMAGE PATH HERE
if __name__ == "__main__":
    # ==================================================================
    # ADD YOUR IMAGE PATH HERE (replace 'path_to_your_image.jpg')
    image_path = 'test2.jpeg'  
    # ==================================================================
    
    if os.path.exists(image_path):
        predict_image_class(image_path)
    else:
        print(f"Error: Image not found at {image_path}")
        print("Please make sure to:")
        print("1. Replace 'path_to_your_image.jpg' with your actual image path")
        print("2. Ensure the image exists at that location")