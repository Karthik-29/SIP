import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the data from CSV
df = pd.read_csv("result.csv")  # Ensure the CSV has columns: 'day', 'Return'

# Extract day and return
days = df["Day"].values
returns = df["Return"].values

# Reshape the data for polynomial fitting
days_reshaped = days.reshape(-1, 1)

# Function to fit polynomial models and calculate MSE
def fit_polynomial(degree):
    poly = PolynomialFeatures(degree=degree)
    days_poly = poly.fit_transform(days_reshaped)
    
    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(days_poly, returns)
    
    # Predict and calculate MSE
    predictions = model.predict(days_poly)
    mse = mean_squared_error(returns, predictions)
    return model, mse, predictions

# Fit and evaluate polynomial models of degree 2, 3, and 4
results = {}
for degree in range(2, 5):
    model, mse, predictions = fit_polynomial(degree)
    results[degree] = {'model': model, 'mse': mse, 'predictions': predictions}
    print(f"Degree {degree}: MSE = {mse:.4f}")

# Find the best polynomial model based on MSE
best_degree = min(results, key=lambda x: results[x]['mse'])
best_model = results[best_degree]['model']
best_predictions = results[best_degree]['predictions']

# Plot the data and the best fit polynomial
plt.scatter(days, returns, color="blue", label="Data")
plt.plot(days, best_predictions, color="red", label=f"Best Fit (Degree {best_degree})")
plt.xlabel("day")
plt.ylabel("Return")
plt.title(f"Polynomial Fit (Degree {best_degree}) with MSE = {results[best_degree]['mse']:.4f}")
plt.legend()
plt.grid(True)
plt.show()
