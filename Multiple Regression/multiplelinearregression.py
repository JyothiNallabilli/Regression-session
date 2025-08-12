import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your CSV file
df = pd.read_csv("sampledata.csv")

# Step 2: Define features and target
X = df[['size', 'bedrooms', 'age']]  # Adjust column names if needed
y = df['price']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on both sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step 6: Evaluate training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Step 7: Evaluate test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Step 8: Print results
print("ðŸ“Š Training Set Evaluation:")
print(f"Training MSE: {train_mse:.2f}")
print(f"Training RÂ² Score: {train_r2:.2f}")

print("\nðŸ“Š Test Set Evaluation:")
print(f"Test MSE: {test_mse:.2f}")
print(f"Test RÂ² Score: {test_r2:.2f}")

# Step 9: Visualize best fit line (test set)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit Line')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Best Fit Line on Test Set (RÂ² = {test_r2:.2f})")
plt.legend()
plt.grid(True)
plt.show()
