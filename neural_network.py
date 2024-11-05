


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Load and prepare the data
df = pd.read_csv("/home/mlahlbac/COURSES/Geophysical_data_science/Tuddal_data.csv", na_values=np.nan)

df["albedo"] = df['R_SW_in'] / df['R_LW_out_corr']

# Define features and target variable
features = ["wind_speed", "max_wind_speed", "R_SW_in", "precip_int_h_D", "albedo",
        "air_temperature", "wind_dir", "specific_humidity",
        "air_pressure", "precip_int_h_D", 
        "FC1DRIFTsum_99_99_1_1_1", "FC2DRIFTsum_99_99_1_1_1"]

df = df.dropna(subset=['LE_qc0']) 
df = df.dropna(subset=features) 

y = df["LE_qc0"]
X = df[features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of units in the layers
units = 64

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Setting input shape
model.add(Dense(
    units,
    activation="relu",
    kernel_initializer="uniform"
))
model.add(Dropout(0.2))  # Add dropout to prevent overfitting

# Add more layers if necessary
model.add(Dense(units, activation="relu"))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(1))  # Output layer for regression (single output)

# Print a summary of the model
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

######################################################################
# PLOT RESULTS:

import matplotlib.pyplot as plt

# Train the model and store the training history
history = model.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.2)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Predict the target values for the test set
y_pred = model.predict(X_test)

# Scatter plot of predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Calculate residuals
residuals = y_test - y_pred.reshape(-1)

# Histogram of residuals
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=50, alpha=0.75)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Plot residuals to check for randomness (should be centered around 0)
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r')
plt.title('Residuals Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

################################################
# FEATURE IMPORTANCE -- PERMUTATION IMPORTANCE #

import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Assuming you have trained your model and split your data into X_train, X_test, y_train, y_test

# Evaluating feature importance using permutation importance
# Neural networks from Keras can be wrapped in a sklearn-compatible way using KerasRegressor
from scikeras.wrappers import KerasClassifier, KerasRegressor

def create_model():
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu', kernel_initializer='uniform'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

keras_model = KerasRegressor(model=create_model, epochs=100, batch_size=10, verbose=0)
keras_model.fit(X_train, y_train)

# Compute permutation feature importance
result = permutation_importance(keras_model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

# Summarize feature importance
sorted_idx = result.importances_mean.argsort()
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], xerr=result.importances_std[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title("Permutation Feature Importance")
plt.xlabel("Mean Decrease in Impurity")
plt.show()
