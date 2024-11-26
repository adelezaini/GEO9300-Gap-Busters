
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Import regression metrics
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("../data/Tuddal_data.csv", na_values=np.nan)

# Drop rows with NaN values in the 'LE_qc0' column
df = df.dropna(subset=['LE_qc0']) 

# Define target and feature variables
y = df["LE_qc0"]
X = df[["wind_speed", 
        "R_SW_in", 
#        "albedo", 
        "precip_int_h_D",
        "air_temperature",
        "wind_dir",
        "R_LW_out_corr",
        "specific_humidity", 
        "air_pressure", 
        "FC1DRIFTsum_99_99_1_1_1", 
        "FC2DRIFTsum_99_99_1_1_1"]]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Performance statistics:
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2:.2f}')


# Plotting True vs Predicted values
plt.figure(figsize=(10, 5))

# Scatter plot for true vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")

plt.tight_layout()
plt.show()

###

