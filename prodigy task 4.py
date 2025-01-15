import warnings
warnings.filterwarnings('ignore')

import keras
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random

# Categories and Image Size
CATEGORIES = ["01_palm", '02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']
IMG_SIZE = 50

# Paths for dataset (Update this path with your local dataset directory)
data_path = "C:\\Users\\rouna\\Downloads\\archive (2).zip"


# Load Images and Classes
image_data = []
for dr in os.listdir(data_path):
    for category in CATEGORIES:
        class_index = CATEGORIES.index(category)
        path = os.path.join(data_path, dr, category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image_data.append([cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)), class_index])
            except Exception as e:
                print(f"Error loading image {img}: {e}")

# Shuffle Data
random.shuffle(image_data)
input_data = []
label = []
for X, y in image_data:
    input_data.append(X)
    label.append(y)

# Visualization
plt.figure(1, figsize=(10, 10))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(image_data[i][0], cmap='hot')
    plt.xticks([])
    plt.yticks([])
    plt.title(CATEGORIES[label[i]][3:])
plt.show()

# Normalize Data
input_data = np.array(input_data) / 255.0
label = np.array(label)

# One Hot Encoding
label = keras.utils.to_categorical(label, num_classes=10, dtype='i1')

# Reshape Data
input_data.shape = (-1, IMG_SIZE, IMG_SIZE, 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.3, random_state=0)

# Model Creation
model = keras.models.Sequential([
    Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and Train the Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))

# Model Summary
model.summary()

# Plot Loss and Accuracy
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
plt.figure(figsize=(10, 10))
sn.heatmap(cm, annot=True, xticklabels=[c[3:] for c in CATEGORIES], yticklabels=[c[3:] for c in CATEGORIES])
plt.show()
