import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the training data
training_file_path = 'training_set.csv'
training_data = pd.read_csv(training_file_path)

# Load the testing data (if available)
testing_file_path = 'testing_set.csv'
testing_data = pd.read_csv(testing_file_path)

# Preliminary data inspection
print(training_data.head())

# Replace 'inf' values with NaN
training_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop irrelevant columns before imputation
columns_to_drop = ['Unnamed: 0', 'pad_id'] # Modify as needed
training_data.drop(columns_to_drop, axis=1, inplace=True)

# Separate the target variable and features for training data
X_train = training_data.drop('OilPeakRate', axis=1)
y_train = training_data['OilPeakRate']

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

# Prepare testing set similarly if available
X_test = testing_data.drop('OilPeakRate', axis=1)
y_test = testing_data['OilPeakRate']
X_test_preprocessed = preprocessor.transform(X_test) # Use transform here, not fit_transform

# Model training
linear_model = LinearRegression()
linear_model.fit(X_train_preprocessed, y_train)

# Model evaluation on the training set
y_pred_train = linear_model.predict(X_train_preprocessed)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
print(f"Training RMSE: {rmse_train}, MAE: {mae_train}, R2: {r2_train}")

# Model evaluation on the testing set (if available)
y_pred_test = linear_model.predict(X_test_preprocessed)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"Testing RMSE: {rmse_test}, MAE: {mae_test}, R2: {r2_test}")
