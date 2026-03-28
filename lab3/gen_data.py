import numpy as np
import matplotlib.pyplot as plt
import pandas as pd   # optional, but makes saving easy

# 1. Generate input data
np.random.seed(42)
n_samples = 200
x = np.random.uniform(-3, 3, n_samples)

# 2. Define a non-linear function (true underlying relationship)
def true_function(x):
    return 0.5 * x**3 - 1.5 * x**2 + 2 * np.sin(2 * x) + 2

# 3. Add Gaussian noise
noise = np.random.normal(0, 0.8, n_samples)
y = true_function(x) + noise

# 4. Create a DataFrame (optional, but convenient for CSV)
df = pd.DataFrame({
    'x': x,
    'y_true': true_function(x),   # you can save the noise‑free target if you like
    'y_noisy': y                  # the actual observations
})

# 5. Save to CSV
df.to_csv('nonlinear_regression_data.csv', index=False)
print("Data saved to 'nonlinear_regression_data.csv'")

# 6. (Optional) Visualize
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y_noisy'], alpha=0.6, label='Observed data', color='blue')
x_dense = np.linspace(-3, 3, 500)
plt.plot(x_dense, true_function(x_dense), 'r-', linewidth=2, label='True function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Non-linear Regression Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()