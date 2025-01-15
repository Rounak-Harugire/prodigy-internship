import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Path to your CSV and images
csv_file_path = "C:\\Users\\rouna\\Downloads\\indian_food.csv"
image_folder_path = "C:\\Users\\rouna\\Downloads\\food_images"  # Replace with your images folder path

# Load the dataset
food_data = pd.read_csv(csv_file_path)

# Display the columns of the dataset
print("Columns in the dataset:", food_data.columns)

# Check the first few rows of the dataset
print(food_data.head())

# Prepare the data for calorie prediction (based on food name and calories)
calories_data = food_data[['name', 'calories']].dropna()
food_calories_map = dict(zip(calories_data['name'], calories_data['calories']))

# Image Classification - Setup
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    image_folder_path,  # Folder with images
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    image_folder_path,  # Folder with images
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='validation'
)

# Load pre-trained ResNet50 model (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Build the custom model
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Calorie prediction function
def predict_calories(food_name):
    return food_calories_map.get(food_name, 'Unknown food item')

# Function to classify image and predict calories
def predict_food_and_calories(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict food class using the image classification model
    food_class_prob = model.predict(img_array)
    food_class_idx = np.argmax(food_class_prob)
    food_class = list(train_generator.class_indices.keys())[food_class_idx]
    
    # Get the calorie prediction for the identified food
    calorie_estimate = predict_calories(food_class)
    
    return food_class, calorie_estimate

# Example usage
image_path = "C:\\Users\\rouna\\Downloads\\food_image.jpg"  # Replace with actual image path
food_class, calorie_estimate = predict_food_and_calories(image_path)

print(f"Predicted Food Class: {food_class}")
print(f"Estimated Calories: {calorie_estimate}")
