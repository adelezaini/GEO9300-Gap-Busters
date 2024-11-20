# LONG SHORT-TERM MEMORY (LSTM) MODEL

# packages:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Function to build the LSTM model (needed for KerasRegressor)
def build_lstm_model(units=50):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Hyperparameters grid to tune
param_grid = {
    'units': [50, 100],  # Number of LSTM units
    'batch_size': [16, 32],
    'epochs': [20, 50]
}

# Wrap the Keras model for use with GridSearchCV
model = KerasRegressor(build_fn=build_lstm_model, verbose=0)

# Perform model tuning and prediction
LSTM_best_model, y_pred = model_tuning_CV(model, param_grid, cv=3, scoring='neg_mean_squared_error')