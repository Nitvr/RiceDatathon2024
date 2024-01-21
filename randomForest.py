import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

# Function to preprocess the dataset
def preprocess_data(data, categorical_cols, target_col='OilPeakRate'):
    y = data[target_col]
    X = data.drop(target_col, axis=1)

    # Apply one-hot encoding to categorical columns
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_encoded = pd.DataFrame(onehot_encoder.fit_transform(X[categorical_cols]))

    # Create feature names for the encoded columns
    encoded_cols = [str(col) + '_' + str(val) for col, vals in zip(categorical_cols, onehot_encoder.categories_) for val in vals]
    X_encoded.columns = encoded_cols
    X_encoded.index = X.index

    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Convert all feature names to strings
    X.columns = X.columns.astype(str)

    return X, y

# Load the training and testing datasets
training_file_path = 'training_set.csv'  # Replace with the path to your training dataset
testing_file_path = 'testing_set.csv'    # Replace with the path to your testing dataset

training_data = pd.read_csv(training_file_path)
testing_data = pd.read_csv(testing_file_path)

# Identify categorical columns (assuming they are the same for both datasets)
categorical_cols = training_data.select_dtypes(include=['object']).columns

# Preprocess the training and testing datasets
X_train, y_train = preprocess_data(training_data, categorical_cols, target_col='OilPeakRate')
X_test, y_test = preprocess_data(testing_data, categorical_cols, target_col='OilPeakRate')

# Parameters for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=0)

# Grid search of parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Use the best model found by grid search
best_rf_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_rf_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
