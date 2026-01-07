import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime
import joblib
import os
import argparse
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

def load_model_and_artifacts(artifacts_dir: str):
    """
    Loads the trained model, project details, and categorical dtypes from disk.

    Args:
        artifacts_dir: The directory where artifacts are saved.

    Returns:
        A tuple containing the loaded model, project details DataFrame, and dtypes dictionary.
    """
    model_path = os.path.join(artifacts_dir, 'xgb_price_model.joblib')
    details_path = os.path.join(artifacts_dir, 'project_details.csv')

    model = joblib.load(model_path)
    project_details_df = pd.read_csv(details_path)

    return model, project_details_df

def predict_price(model: xgb.XGBRegressor, project_details_df: pd.DataFrame, project: str, sqft: float, floorRange: str) -> float:
    """
    Predicts the price for a given property using a loaded XGBoost model.

    Args:
        model: The loaded XGBoost regressor model.
        project_details_df: DataFrame with project-specific information.
        project: The name of the project.
        sqft: The square footage of the property.
        floorRange: The floor range of the property (e.g., '06-10').

    Returns:
        The predicted price.
    """
    today = datetime.now()
    prediction_year = today.year
    prediction_month = today.month

    # Fetch project-specific data
    project_data = project_details_df[project_details_df['project'] == project].iloc[0]
    district = project_data['district']
    market_segment = project_data['marketSegment']
    start_year = project_data['startYear']
    clean_tenure = project_data['cleanTenure']

    # Calculate 'Age' and 'remainingLease'
    age = np.ceil(today.year - start_year)
    remaining_lease = clean_tenure - age

    # Create a DataFrame for prediction
    prediction_data = pd.DataFrame({
        'district': [district],
        'floorRange': [floorRange],
        'marketSegment': [market_segment],
        'project': [project],
        'cleanTenure': [clean_tenure],
        'startYear': [start_year],
        'Age': [age],
        'remainingLease': [remaining_lease],
        'sqft': [sqft],
        'year': [prediction_year],
        'month': [prediction_month]
    })

    # Cast categorical columns to 'category' type, matching the training script
    for col in ['district', 'floorRange', 'marketSegment', 'project', 'cleanTenure']:
        prediction_data[col] = prediction_data[col].astype('category')

    # Predict the price
    predicted_price = model.predict(prediction_data)

    return predicted_price[0]

def analyze_and_predict_trend(full_data_path: str, project: str, floorRange: str, sqft: float, current_predicted_price: float):
    """
    Filters historical data, fits a polynomial curve of price vs. date,
    and predicts future price trends.

    Args:
        full_data_path: Path to the complete transaction CSV file.
        project: The name of the project to filter by.
        floorRange: The floor range to filter by.
        sqft: The square footage to base the trend analysis around.
        current_predicted_price: The price predicted for today by the XGBoost model.
    """
    print("\n--- Historical Trend Analysis ---")
    try:
        df = pd.read_csv(full_data_path)
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print(f"Error: The full data file was not found at '{full_data_path}'. Skipping trend analysis.")
        return

    # Filter data for the specific project and floor range
    sqft_min, sqft_max = sqft - 50, sqft + 50
    filtered_df = df[(df['project'] == project) & (df['floorRange'] == floorRange) & (df['sqft'].between(sqft_min, sqft_max))].copy()

    if len(filtered_df) < 3:
        print(f"Insufficient historical data for '{project}' (floor range '{floorRange}', sqft {sqft_min}-{sqft_max}) to perform trend analysis (found {len(filtered_df)} records).")
        return

    # Prepare data for polynomial fitting
    filtered_df['date_ordinal'] = filtered_df['Date'].map(datetime.toordinal)
    x = filtered_df['date_ordinal']
    y = filtered_df['price']

    # Automatically determine the best polynomial degree using AIC
    best_degree = -1
    min_aic = float('inf')

    try:
        for degree in range(1, 6): # Check polynomial degrees from 1 to 5
            if len(filtered_df) <= degree:
                break # Cannot fit a polynomial of degree >= number of data points
            
            p, raw_residuals, _, _, _ = np.polyfit(x, y, degree, full=True)

            if isinstance(raw_residuals, list) and raw_residuals: # If it's a non-empty list
                rss = raw_residuals[0]
            elif isinstance(raw_residuals, np.ndarray) and raw_residuals.size > 0: # If it's a non-empty array
                rss = raw_residuals.sum()
            else:
                rss = 0
            
            # Calculate AIC (Akaike Information Criterion)
            n = len(y)
            k = degree + 1
            if rss > 0:
                aic = n * np.log(rss / n) + 2 * k
                if aic < min_aic:
                    min_aic = aic
                    best_degree = degree
    except:
        best_degree == -1

    today = datetime.now()
    if best_degree == -1:
        print("Could not determine a suitable trend line for the historical data.")
    else:
        print(f"Found best-fit polynomial of degree: {best_degree}")
        best_fit_poly = np.poly1d(np.polyfit(x, y, best_degree))

        # Predict prices for future dates
        future_dates = {
            "3 months": today + relativedelta(months=3),
            "1 year": today + relativedelta(years=1),
            "3 years": today + relativedelta(years=3),
            "5 years": today + relativedelta(years=5)
        }

        print("\nFuture Price Projections based on Historical Trend:")
        for label, future_date in future_dates.items():
            future_date_ordinal = future_date.toordinal()
            predicted_price = best_fit_poly(future_date_ordinal)
            print(f"  - In {label} ({future_date.strftime('%Y-%m-%d')}): ${predicted_price:,.2f}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    # Plot historical data
    plt.scatter(filtered_df['Date'], filtered_df['price'], label='Historical Transactions', color='skyblue', alpha=0.8, edgecolors='b')

    # Add sqft labels to each historical point
    for _, row in filtered_df.iterrows():
        plt.text(row['Date'], row['price'], f" {row['sqft']:.0f}", fontsize=8, verticalalignment='bottom', ha='center')

    # Plot the fitted trend line
    plot_start_date = filtered_df['Date'].min()
    if best_degree != -1:
        plot_end_date = max(future_dates.values())
        plot_dates = pd.date_range(start=plot_start_date, end=plot_end_date, periods=300)
        plot_dates_ordinal = [d.toordinal() for d in plot_dates]
        curve_prices = best_fit_poly(plot_dates_ordinal)
        plt.plot(plot_dates, curve_prices, label=f'Best-Fit Trend (Degree {best_degree})', color='red', linestyle='--')

    # Plot the current XGBoost prediction
    plt.scatter([today], [current_predicted_price], label=f'Current XGBoost Prediction: ${current_predicted_price:,.0f}', color='green', s=150, zorder=5, marker='*')

    if best_degree != -1:
        # Plot future projections
        future_plot_dates = list(future_dates.values())
        future_plot_prices = [best_fit_poly(d.toordinal()) for d in future_plot_dates]
        plt.scatter(future_plot_dates, future_plot_prices, label='Future Trend Projections', color='purple', s=120, zorder=5, marker='X')

    plt.title(f"Price Analysis and Trend Projection for '{project}'\n(Floor: {floorRange}, Sqft: {sqft:.0f} +/- 50)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Set up argument parser to accept user inputs from the command line
    parser = argparse.ArgumentParser(description='Predict property price using a trained XGBoost model.')
    parser.add_argument('--project', type=str, default='THE ANCHORAGE', 
                        help='Name of the project (default: THE ANCHORAGE)')
    parser.add_argument('--sqft', type=float, default=1200, 
                        help='Square footage of the unit (default: 1200)')
    parser.add_argument('--floor_range', type=str, default='06-10', 
                        help='Floor range of the unit, e.g., "06-10"\n Allowed values: "-", "01-05", "06-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "B1-B5"')

    args = parser.parse_args()

    # 1. Load the model and artifacts
    artifacts_path = 'artifacts'
    model, details_df = load_model_and_artifacts(artifacts_path)
    print("Model and artifacts loaded successfully.")

    # 2. Predict the price using inputs from the command line or default values
    price = predict_price(model, details_df, args.project, args.sqft, args.floor_range)
    print(f"\nPredicted price for today for a {args.sqft} sqft unit in '{args.project}' on floor range {args.floor_range}: ${price:,.2f}")

    # 3. Perform trend analysis
    full_data_file = 'data/PMI_Res_Transactions_20251230.csv'
    analyze_and_predict_trend(full_data_file, args.project, args.floor_range, args.sqft, price)