import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv(r"C:\Users\rouna\Downloads\house_prices_dataset.csv")

# Print column names to check available columns
print("Dataset Columns:", dataset.columns)

# Use the correct column name 'Housesize(sqft)' for feature and 'Price' for target
X = dataset[['Housesize(sqft)']]  # Use 'Housesize(sqft)' instead of 'Square_Feet'
y = dataset['Price']  # 'Price' is correct for the target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the sizes of X_train and y_train
print("Sizes of X_train and y_train:", len(X_train), len(y_train))

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the training data and the regression line
plt.scatter(X_train, y_train, color='red', label='Training data')
plt.plot(X_train, model.predict(X_train), color='blue', label='Regression line')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression on House Prices')
plt.show()

# Print the model's coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model
print("Model R^2 score:", model.score(X_test, y_test))
