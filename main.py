# IMPORT LIBRARIES
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# STEP 1: DATA PREPROCESSING

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
train_path = "dataset/train"
test_path = "dataset/test"
# Training Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Testing Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
print("Data Loaded Successfully")

# STEP 2: BUILD MODEL (TRANSFER LEARNING)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False
# Add custom layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# STEP 3: TRAIN MODEL

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)
# Save model
model.save("damage_model.h5")
print("Model Trained and Saved Successfully")

# STEP 4: EVALUATE MODEL

loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)

# STEP 5: PREDICTION FUNCTION

from tensorflow.keras.preprocessing import image
# Load trained model
model = load_model("damage_model.h5")
# Class labels
classes = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Prediction
    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]
    print("Predicted Damage Level:", result)

# STEP 6: TEST WITH SAMPLE IMAGE

test_image = "test.jpg"   # Give your test image path here
predict_image(test_image)


