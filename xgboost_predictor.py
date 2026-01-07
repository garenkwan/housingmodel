import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import joblib
import os
 
def train_price_prediction_model(file_path: str):
    """
    Loads transaction data, preprocesses it, and trains an XGBoost model for price prediction.
    It uses GridSearchCV for hyperparameter tuning and cross-validation.

    Args:
        file_path: The path to the CSV file containing transaction data.

    Returns:
        A tuple containing the retrained XGBoost model, a dataframe with unique project details, and a dictionary of categorical dtypes.
    """
    df = pd.read_csv(file_path)

    # Convert 'Date' to datetime and extract numerical features
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month

    # Explicitly cast categorical columns
    for col in ['district', 'floorRange', 'marketSegment', 'project', 'cleanTenure']:
        df[col] = df[col].astype('category')

    # Feature selection - updated to include year and month, and remove Date
    features = ['district', 'floorRange', 'marketSegment', 'project', 'cleanTenure', 'startYear', 'Age', 'remainingLease', 'sqft', 'year', 'month']
    target = 'price'

    X = df[features]
    y = df[target]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the XGBoost model
    # We use 'reg:absoluteerror' as the objective and 'neg_mean_absolute_percentage_error' for scoring
    xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', enable_categorical=True, random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring='neg_mean_absolute_percentage_error',
                               cv=5, verbose=1, n_jobs=-1)

    print("Starting hyperparameter tuning with cross-validation...")
    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")

    # Retrain the model on the entire dataset with the best parameters
    print("Retraining model with best parameters on the full dataset...")
    model = grid_search.best_estimator_

    # Create a separate dataframe for project-specific lookups to be used in prediction
    project_details_cols = ['project', 'district', 'marketSegment', 'startYear', 'cleanTenure']
    project_details_df = df[project_details_cols].drop_duplicates(subset=['project']).reset_index(drop=True)

    return model, project_details_df

if __name__ == '__main__':
    # Define file paths
    file_name = 'data/PMI_Res_Transactions_20251230.csv'
    output_dir = 'artifacts'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'xgb_price_model.joblib')
    details_path = os.path.join(output_dir, 'project_details.csv')

    # 1. Train the model
    trained_model, project_details = train_price_prediction_model(file_name)
    print("\nModel training complete.")

    # 2. Save the model and supporting artifacts
    joblib.dump(trained_model, model_path)
    project_details.to_csv(details_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"Project details saved to {details_path}")
