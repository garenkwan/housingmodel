import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_hdb_resale_model(file_path: str):
    """
    Loads HDB resale data, preprocesses it, and trains an XGBoost model for price prediction.
    It uses GridSearchCV for hyperparameter tuning and cross-validation.

    Args:
        file_path: The path to the CSV file containing HDB resale transaction data.

    Returns:
        A tuple containing the retrained XGBoost model and a dataframe with unique flat details.
    """
    df = pd.read_csv(file_path)

    # Feature Engineering
    df['year'] = pd.to_numeric(df['month'].apply(lambda x: x.split('-')[0]))
    df['month'] = pd.to_numeric(df['month'].apply(lambda x: x.split('-')[1]))
    df['lease_left'] = 99 - (df['year'] - df['lease_commence_date'])

    # Explicitly cast categorical columns
    categorical_features = ['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model']
    for col in categorical_features:
        df[col] = df[col].astype('category')

    # Feature selection
    features = [
        'year', 'month', 'town', 'flat_type', 'block', 'street_name',
        'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'lease_left'
    ]
    target = 'resale_price'

    X = df[features]
    y = df[target]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the XGBoost model with support for categorical features
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True, random_state=42)

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

    # Create a separate dataframe for flat-specific lookups to be used in prediction
    flat_details_cols = ['town', 'block', 'street_name', 'flat_model', 'lease_commence_date', 'flat_type', 'storey_range']
    # We drop duplicates across all identifying columns to create a clean lookup table.
    flat_details_df = df[flat_details_cols].drop_duplicates(subset=flat_details_cols).reset_index(drop=True)

    return model, flat_details_df

if __name__ == '__main__':
    # Define file paths
    file_name = 'data/ResaleflatpricesJan2017onwards-20260107.csv'
    output_dir = 'artifacts'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'xgb_hdb_resale_model.joblib')
    details_path = os.path.join(output_dir, 'flat_details.csv')

    # 1. Train the model
    print(f"Training model with data from {file_name}...")
    trained_model, flat_details = train_hdb_resale_model(file_name)
    print("\nModel training complete.")

    # 2. Save the model and supporting artifacts
    joblib.dump(trained_model, model_path)
    flat_details.to_csv(details_path, index=False)

    print(f"Model saved to {model_path}")
    print(f"Flat details lookup table saved to {details_path}")

    print("\nTo run a prediction, use the following command:")
    print("python predict_hdb_resale_price.py --town \"ANG MO KIO\" --flat_type \"4 ROOM\" --block 310B --street_name \"ANG MO KIO AVE 1\" --storey_range \"01 TO 03\" --floor_area_sqm 92.0")