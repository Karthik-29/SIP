import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Read CSV file into DataFrame
df = pd.read_csv("result.csv")  # Ensure the CSV has columns: 'day', 'Return'

# Extract day and return
days = df["Day"].values
returns = df["Return"].values

# Pearson Correlation
pearson_r, pearson_p = stats.pearsonr(days, returns)
print(f"Pearson correlation: {pearson_r:.4f}, p-value: {pearson_p:.4f}")

# Spearman Correlation
spearman_r, spearman_p = stats.spearmanr(days, returns)
print(f"Spearman correlation: {spearman_r:.4f}, p-value: {spearman_p:.4f}")

# Linear Regression
slope, intercept, r_value, p_value, std_err = stats.linregress(days, returns)
print(f"Linear Regression: Slope={slope:.4f}, Intercept={intercept:.4f}, R²={r_value**2:.4f}, p-value={p_value:.4f}")

# Plot regression line
plt.scatter(days, returns, color="blue", alpha=0.7, label="Data")
plt.plot(days, slope * days + intercept, color="red", label="Regression Line")
plt.xlabel("day")
plt.ylabel("Return")
plt.title("Linear Regression: Day vs. Return")
plt.legend()
plt.grid(True)
plt.show()
