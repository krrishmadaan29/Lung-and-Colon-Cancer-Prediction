import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch  # Updated import
from keras_tuner.engine.hyperparameters import HyperParameters  # Updated import
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)   

# Dataset paths
data_dir = 'lung_colon_image_set'
IMG_SIZE = (227, 227)  # AlexNet input size
BATCH_SIZE = 32
EPOCHS = 30

## Image Preprocessing Functions
def enhance_image(image):
    # Contrast Limited Adaptive Histogram Equalization
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def preprocess_image(image):
    # Resize
    image = cv2.resize(image, IMG_SIZE)
    # Enhance
    image = enhance_image(image)
    # Scale to [0,1]
    image = image / 255.0
    # Normalize
    image = (image - np.mean(image)) / np.std(image)
    return image

# Enhanced Data preparation function with detailed printing
def prepare_data(data_dir):
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    classes = []
    data = []
    class_counts = {}
    
    for tissue_type in ['colon_image_sets', 'lung_image_sets']:
        tissue_path = os.path.join(data_dir, tissue_type)
        print(f"\nProcessing {tissue_type.replace('_', ' ')}:")
        
        for class_name in os.listdir(tissue_path):
            class_path = os.path.join(tissue_path, class_name)
            if os.path.isdir(class_path):
                full_class_name = f"{tissue_type[:5]}_{class_name}"
                class_images = [f for f in os.listdir(class_path) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))]
                num_images = len(class_images)
                class_counts[full_class_name] = num_images
                classes.append(full_class_name)
                
                print(f"  - Class '{full_class_name}': {num_images} images")
                
                for img_name in class_images:
                    img_path = os.path.join(class_path, img_name)
                    data.append({'path': img_path, 'label': full_class_name})
    
    df = pd.DataFrame(data)
    
    # Print total dataset statistics
    total_images = len(df)
    print("\n" + "-"*50)
    print(f"TOTAL DATASET: {total_images} images across {len(class_counts)} classes")
    print("-"*50)
    
    # Split into train and test with stratification
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['label'])  # 0.25 x 0.8 = 0.2
    
    # Print split information
    print("\nDATA SPLITS:")
    print(f"Training images:   {len(train_df)} ({len(train_df)/total_images*100:.1f}%)")
    print(f"Validation images: {len(val_df)} ({len(val_df)/total_images*100:.1f}%)")
    print(f"Test images:       {len(test_df)} ({len(test_df)/total_images*100:.1f}%)")
    
    # Print class distribution in each split
    print("\nCLASS DISTRIBUTION:")
    print("{:<20} {:<10} {:<10} {:<10}".format('Class', 'Train', 'Val', 'Test'))
    for cls in sorted(class_counts.keys()):
        train_count = sum(train_df['label'] == cls)
        val_count = sum(val_df['label'] == cls)
        test_count = sum(test_df['label'] == cls)
        print("{:<20} {:<10} {:<10} {:<10}".format(
            cls, train_count, val_count, test_count))
    
    print("="*50 + "\n")
    
    return train_df, val_df, test_df, sorted(set(classes))

# Prepare data with detailed printing
train_df, val_df, test_df, classes = prepare_data(data_dir)

# Custom data generator with preprocessing
def custom_generator(df, batch_size, augment=False):
    while True:
        batch = df.sample(n=batch_size)
        images = []
        labels = []
        
        for _, row in batch.iterrows():
            # Read and preprocess image
            img = cv2.imread(row['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_image(img)
            
            # One-hot encode label
            label = tf.keras.utils.to_categorical(classes.index(row['label']), num_classes=len(classes))
            
            images.append(img)
            labels.append(label)
        
        yield np.array(images), np.array(labels)

# Create data generators
train_gen = custom_generator(train_df, BATCH_SIZE, augment=True)
val_gen = custom_generator(val_df, BATCH_SIZE)  # Use the explicit validation set
test_gen = custom_generator(test_df, BATCH_SIZE)

# AlexNet model builder with hyperparameter tuning
def build_model(hp):
    model = models.Sequential()
    
    # 1st Convolutional Layer
    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', 64, 96, step=16),
        kernel_size=(11, 11),
        strides=(4, 4),
        activation='relu',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', 128, 384, step=64),
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())
    
    # 3rd-5th Convolutional Layers
    for i in range(3, 6):
        model.add(layers.Conv2D(
            filters=hp.Int(f'conv{i}_filters', 192, 384, step=64),
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ))
        if i == 5:
            model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(layers.BatchNormalization())
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    
    for i in range(1, 3):
        model.add(layers.Dense(
            units=hp.Int(f'dense{i}_units', 256, 4096, step=256),
            activation='relu'
        ))
        model.add(layers.Dropout(hp.Float(f'dropout{i}', 0.3, 0.7, step=0.1)))
    
    # Output Layer
    model.add(layers.Dense(len(classes), activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(
            hp.Float('learning_rate', 1e-5, 1e-3, sampling='LOG')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=1,
    directory='alexnet_tuning',
    project_name='lung_colon_cancer'
)

# Callbacks
callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint('best_alexnet_model.h5', save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
]

# Hyperparameter search
tuner.search(
    train_gen,
    steps_per_epoch=len(train_df)//BATCH_SIZE,
    epochs=20,
    validation_data=val_gen,
    validation_steps=len(train_df)*0.2//BATCH_SIZE,
    callbacks=callbacks
)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train best model
history = best_model.fit(
    train_gen,
    steps_per_epoch=len(train_df)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(train_df)*0.2//BATCH_SIZE,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = best_model.evaluate(test_gen, steps=len(test_df)//BATCH_SIZE)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save final model
best_model.save('alexnet_lung_colon_final.h5')

# Generate predictions
y_true = []
y_pred = []
for _ in range(len(test_df)//BATCH_SIZE):
    x, y = next(test_gen)
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(best_model.predict(x), axis=1))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, 
            yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()