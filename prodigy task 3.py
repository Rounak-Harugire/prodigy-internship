import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Directory containing image data
dir = "C:\\Users\\rouna\\Downloads\\archive (1).zip"
categories = ['Cat', 'Dog']

# Prepare data
data = []
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)
        try:
            pet_img = cv2.resize(pet_img, (50, 50))
            image = np.array(pet_img).flatten()
            data.append([image, label])
        except Exception as e:
            print(f"Error processing image {imgpath}: {e}")
            pass

# Save data to a pickle file
with open('data1.pickle', 'wb') as pick_in:
    pickle.dump(data, pick_in)

# Load data from pickle file
with open('data1.pickle', 'rb') as pick_in:
    data = pickle.load(pick_in)

# Shuffle and prepare features and labels
random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.01)

# Train model (commented out as the model file is already being used)
# model = SVC(C=1, kernel='poly', gamma='auto')
# model.fit(xtrain, ytrain)

# Save the trained model
# with open('model.sav', 'wb') as pick_out:
#     pickle.dump(model, pick_out)

# Load the trained model
with open('model.sav', 'rb') as pick:
    model = pickle.load(pick)

# Make predictions and calculate accuracy
prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

# Display results
categories = ['Cat', 'Dog']
print('Accuracy is:', accuracy)
print('Prediction is:', categories[prediction[0]])

mypet = xtest[0].reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.show()
