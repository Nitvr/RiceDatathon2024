import pandas as pd
import numpy as np
from keras.saving.save import load_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load the training and testing data

training_file_path = 'training_set.csv'
testing_file_path = 'testing_set.csv'  # Adjust this to your testing data file name

training_data = pd.read_csv(training_file_path)
testing_data = pd.read_csv(testing_file_path)

# Replace 'inf' values with NaN in both datasets
training_data.replace([np.inf, -np.inf], np.nan, inplace=True)
testing_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop irrelevant columns before imputation in both datasets
columns_to_drop = ['Unnamed: 0', 'pad_id'] # Modify as needed
training_data.drop(columns_to_drop, axis=1, inplace=True)
testing_data.drop(columns_to_drop, axis=1, inplace=True)

# Separate the target variable and features for both datasets
X_train = training_data.drop('OilPeakRate', axis=1)
y_train = training_data['OilPeakRate']

X_test = testing_data.drop('OilPeakRate', axis=1)
y_test = testing_data['OilPeakRate']

# Identifying categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

# Creating a ColumnTransformer for transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Applying the transformations to the training features
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply the same transformations to the testing features
X_test_preprocessed = preprocessor.transform(X_test)

# Define L1 and L2 regularization factors
l1_factor = 0.01  # Adjust as needed
l2_factor = 0.01  # Adjust as needed

# Build the FNN model with more layers and L1/L2 regularization
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_preprocessed.shape[1],),
          kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
    Dropout(0.5),
    Dense(256, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
    Dropout(0.4),
    Dense(128, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
    Dropout(0.3),
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
    Dropout(0.2),
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)),
    Dropout(0.1),
    Dense(1, kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor))  # Output layer for regression
])

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch % 50 == 0 and epoch != 0:
        return lr * 0.98  # Reduce the learning rate by half every 50 epochs
    return lr

callback = LearningRateScheduler(scheduler)

# Compile the model with an initial learning rate
initial_learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss='mse')

# Train the model with the learning rate scheduler
history = model.fit(X_train_preprocessed, y_train, epochs=10000, batch_size=128, validation_split=0.2, verbose=2, callbacks=[callback])

# Save the trained model
model.save('FNN_model.h5')

# Load the saved model
saved_model = load_model('FNN_model.h5')

# Predict and evaluate the model on the test set
y_pred = saved_model.predict(X_test_preprocessed)


print(y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE: ", rmse)
