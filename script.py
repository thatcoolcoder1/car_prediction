

import numpy as np
import pandas as pd
import matplotlib

# Use an interactive backend to open the plot in a separate window
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

# Load the dataset
#data = pd.read_csv("car_data.csv")
data = pd.read("car_data.csv",name=["Year", "Mileage", "sp"])
data = (data - data.mean())/data.std()

# Scale Mileage and selling price for stable gradient descent
x = data["Mileage"].values / 10000
y = data["sp"].values / 10000


# Prediction function
def predict(x, w, b):
    return w * x + b


# Calculate the cost function
def compute_cost(x, y, w, b):
    m = len(x)
    total = 0

    for i in range(m):
        prediction = predict(x[i], w, b)
        error = prediction - y[i]
        total += error**2

    return total / (2 * m)


# Calculate gradients for w and b
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        prediction = predict(x[i], w, b)
        error = prediction - y[i]

        dj_dw += error * x[i]
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# Gradient descent algorithm
def gradient_descent(x, y, w, b, learning_rate, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update parameters
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print("Iteration:", i, "Cost:", cost)

    return w, b


# Initialize parameters
w = 0
b = 0

learning_rate = 0.01
iterations = 10000

# Train the model
w, b = gradient_descent(
    x, y, w, b, learning_rate, iterations
)

print("\nTraining finished!")
print("w =", w)
print("b =", b)
print("Final cost =", compute_cost(x, y, w, b))


# Predict the price for a car with 45,000 km mileage
mileage = 45000

# The model expects scaled mileage
scaled_mileage = mileage / 10000

predicted_scaled_price = predict(scaled_mileage, w, b)

# Convert the prediction back to the original price scale
predicted_price = predicted_scaled_price * 10000

print(
    "\nPredicted price for",
    mileage,
    "km:",
    predicted_price
)


# Plot actual data and the regression line
mileage_data = data["Mileage"].values
price_data = data["sp"].values

# Sort mileage values so the regression line is drawn neatly
sorted_indices = np.argsort(mileage_data)
sorted_mileage = mileage_data[sorted_indices]

# Scale the mileage before passing it to the model
scaled_mileage_data = sorted_mileage / 10000

# Generate predicted prices
predicted_scaled_prices = predict(
    scaled_mileage_data, w, b
)
predicted_prices = predicted_scaled_prices * 10000

# Plot the data points and fitted line
plt.figure(figsize=(10, 6))

plt.scatter(
    mileage_data,
    price_data,
    label="Actual car prices",
    alpha=0.7
)

plt.plot(
    sorted_mileage,
    predicted_prices,
    label="Linear regression line",
    linewidth=2
)

plt.xlabel("Mileage (km)")
plt.ylabel("Selling price")
plt.title("Car Price Prediction Using Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
