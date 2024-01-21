import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Function to preprocess the data
def preprocess_data(data):
    data.drop(columns=['Unnamed: 0'], inplace=True)  # Assuming 'Unnamed: 0' is not needed
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    X = pd.get_dummies(data.drop('OilPeakRate', axis=1))
    y = data['OilPeakRate']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = np.expand_dims(X_scaled, axis=2)
    return X_reshaped, y

# Load the training and testing datasets
training_file_path = 'training_set.csv'  # Replace with your path
testing_file_path = 'testing_set.csv'    # Replace with your path

training_data = pd.read_csv(training_file_path)
testing_data = pd.read_csv(testing_file_path)

# Preprocess the training and testing datasets
X_train_reshaped, y_train = preprocess_data(training_data)
X_test_reshaped, y_test = preprocess_data(testing_data)

# Define the CNN model
model = Sequential([
    Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_reshaped, y_train, epochs=250, batch_size=128, validation_split=0.2, verbose=2, callbacks=[early_stopping])

# Predict on test set
y_pred = model.predict(X_test_reshaped)

# Evaluate the model using RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE: ", rmse)


# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'][5:], label='Train Loss')
plt.plot(history.history['val_loss'][5:], label='Validation Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(False)
plt.savefig('CNNresult.png')
plt.show()

