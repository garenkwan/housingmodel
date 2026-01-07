import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime
import joblib
import os
import argparse
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

def get_hdb_dropdown_values(data_path: str):
    """
    Reads the HDB data file and returns unique values for dropdowns.

    Args:
        data_path (str): Path to the full resale data CSV.

    Returns:
        A tuple of lists: (towns, flat_types, storey_ranges)
    """
    df = pd.read_csv(data_path)
    towns = sorted(df['town'].unique())
    flat_types = sorted(df['flat_type'].unique())
    storey_ranges = sorted(df['storey_range'].unique())
    return towns, flat_types, storey_ranges


def predict_resale_price(town: str, flat_type: str, block: str, street_name: str,
                         storey_range: str, floor_area_sqm: float,
                         model_path: str, data_path: str):
    """
    Predicts the resale price of an HDB flat based on user inputs.

    Args:
        town (str): Town of the flat.
        flat_type (str): The type of the flat.
        block (str): The block number.
        street_name (str): The street name.
        storey_range (str): The storey range of the unit.
        floor_area_sqm (float): The floor area in square meters.
        model_path (str): Path to the trained XGBoost model file.
        data_path (str): Path to the full resale data CSV for detail lookup.

    Returns:
        float: The predicted resale price.
    """
    # Load the trained model and flat details
    model = joblib.load(model_path)
    full_df = pd.read_csv(data_path)

    # Get today's year and month
    today = datetime.now()
    year = today.year
    month = today.month

    # Find matching transactions in the historical data to infer flat_model and lease_commence_date
    block_matches = full_df[
        (full_df['town'] == town) &
        (full_df['block'] == block) &
        (full_df['street_name'] == street_name) &
        (full_df['flat_type'] == flat_type)
    ]

    if block_matches.empty:
        raise ValueError("Could not find any historical transactions for the given town, block, street, and flat type. Cannot infer flat details.")

    # Use the most common (mode) flat_model and lease_commence_date from historical transactions
    # for that specific flat type in the block. This is a robust way to infer the details.
    flat_model = block_matches['flat_model'].mode()[0]
    lease_commence_date = block_matches['lease_commence_date'].mode()[0]

    # Calculate remaining lease
    lease_left = 99 - (year - lease_commence_date)

    # Prepare the feature vector for prediction in the correct order
    features = [
        year, month, town, flat_type, block,
        street_name, storey_range, floor_area_sqm,
        flat_model, lease_commence_date, lease_left
    ]

    feature_names = [
        'year', 'month', 'town', 'flat_type', 'block', 'street_name',
        'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'lease_left'
    ]

    # Create a DataFrame for prediction
    prediction_df = pd.DataFrame([features], columns=feature_names)

    # Ensure categorical types match the training data
    categorical_features = ['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model']
    for col in categorical_features:
        prediction_df[col] = prediction_df[col].astype('category')

    # Make the prediction
    predicted_price = model.predict(prediction_df)

    return predicted_price[0]

def analyze_and_predict_hdb_trend(full_data_path: str, town: str, street_name: str, flat_type: str, storey_range: str, floor_area_sqm: float, current_predicted_price: float):
    """
    Filters historical HDB data, fits a polynomial curve of price vs. date,
    and predicts future price trends, generating a plot.

    Args:
        full_data_path: Path to the complete HDB transaction CSV file.
        town: The town to filter by.
        street_name: The street name to filter by.
        flat_type: The flat type to filter by.
        storey_range: The storey range to filter by.
        floor_area_sqm: The floor area to base the trend analysis around.
        current_predicted_price: The price predicted for today by the XGBoost model.

    Returns:
        A tuple of (matplotlib.figure.Figure, float) or (None, None): The generated plot figure and the trend-based price for today, or None if analysis fails.
    """
    print("\n--- Historical Trend Analysis ---")
    df = pd.read_csv(full_data_path)
    df['date'] = pd.to_datetime(df['month'], format='%Y-%m')

    # Filter data for similar flats
    area_min, area_max = floor_area_sqm - 10, floor_area_sqm + 10
    filtered_df = df[
        (df['town'] == town) &
        (df['street_name'] == street_name) &
        (df['flat_type'] == flat_type) &
        (df['storey_range'] == storey_range) &
        (df['floor_area_sqm'].between(area_min, area_max))
    ].copy()

    if len(filtered_df) < 3:
        print(f"Insufficient historical data for similar flats to perform trend analysis (found {len(filtered_df)} records).")
        return None, None

    # Prepare data for polynomial fitting
    filtered_df['date_ordinal'] = filtered_df['date'].map(datetime.toordinal)
    x = filtered_df['date_ordinal']
    y = filtered_df['resale_price']

    # Fit a polynomial of degree 2 for simplicity with property trends
    degree = 2
    if len(filtered_df) <= degree:
        print("Not enough data points to fit a trend line.")
        return None, None

    try:
        best_fit_poly = np.poly1d(np.polyfit(x, y, degree))
        print(f"Fitted a polynomial trend line of degree: {degree}")
    except Exception as e:
        print(f"Could not fit a trend line due to an error: {e}")
        return None, None

    today = datetime.now()
    today_ordinal = today.toordinal()
    today_poly_price = best_fit_poly(today_ordinal)

    # Predict prices for future dates
    future_dates = {
        "3 months": today + relativedelta(months=3),
        "1 year": today + relativedelta(years=1),
        "3 years": today + relativedelta(years=3),
    }

    print("\nFuture Price Projections based on Historical Trend:")
    for label, future_date in future_dates.items():
        future_date_ordinal = future_date.toordinal()
        predicted_price = best_fit_poly(future_date_ordinal)
        print(f"  - In {label} ({future_date.strftime('%Y-%m-%d')}): ${predicted_price:,.2f}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot historical data
    plt.scatter(filtered_df['date'], filtered_df['resale_price'], label='Historical Transactions', color='skyblue', alpha=0.8, edgecolors='b')

    # Add floor_area_sqm labels to each historical point
    for _, row in filtered_df.iterrows():
        plt.text(row['date'], row['resale_price'], f" {row['floor_area_sqm']:.0f}", fontsize=8, verticalalignment='bottom', ha='center')

    # Plot the fitted trend line
    plot_start_date = filtered_df['date'].min()
    plot_end_date = max(future_dates.values())
    plot_dates = pd.date_range(start=plot_start_date, end=plot_end_date, periods=300)
    plot_dates_ordinal = [d.toordinal() for d in plot_dates]
    curve_prices = best_fit_poly(plot_dates_ordinal)
    plt.plot(plot_dates, curve_prices, label=f'Best-Fit Trend (Degree {degree})', color='red', linestyle='--')

    # Plot the current XGBoost prediction
    plt.scatter([today], [current_predicted_price], label=f'Current XGBoost Prediction: ${current_predicted_price:,.0f}', color='green', s=150, zorder=5, marker='*')

    # Plot the polynomial prediction for today
    plt.scatter([today], [today_poly_price], label=f'Today\'s Trend-Based Price: ${today_poly_price:,.0f}', color='orange', s=150, zorder=5, marker='D')

    # Plot future projections
    future_plot_dates = list(future_dates.values())
    future_plot_prices = [best_fit_poly(d.toordinal()) for d in future_plot_dates]
    plt.scatter(future_plot_dates, future_plot_prices, label='Future Trend Projections', color='purple', s=120, zorder=5, marker='X')

    title = (
        f"Price Analysis and Trend Projection for {flat_type} at {street_name}\n"
        f"Town: {town}, Storey: {storey_range}, Area: {floor_area_sqm:.0f} sqm (±5 sqm)"
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Resale Price ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    return fig, today_poly_price

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict HDB Resale Price.')
    parser.add_argument('--town', type=str, default='ANG MO KIO', help='Town of the flat (e.g., "ANG MO KIO")')
    parser.add_argument('--flat_type', type=str, default='4 ROOM', help='Flat type (e.g., "4 ROOM")')
    parser.add_argument('--block', type=str, default='310B', help='Block number (e.g., 310B)')
    parser.add_argument('--street_name', type=str, default='ANG MO KIO AVE 1', help='Street name (e.g., "ANG MO KIO AVE 1")')
    parser.add_argument('--storey_range', type=str, default='01 TO 03', help='Storey range (e.g., "01 TO 03")')
    parser.add_argument('--floor_area_sqm', type=float, default=92.0, help='Floor area in square meters (e.g., 92.0)')

    args = parser.parse_args()

    # Define artifact paths
    data_file = 'data/ResaleflatpricesJan2017onwards-20260107.csv'
    output_dir = 'artifacts'
    model_file = os.path.join(output_dir, 'xgb_hdb_resale_model.joblib')

    try:
        prediction = predict_resale_price(town=args.town,
                                          flat_type=args.flat_type,
                                          block=args.block,
                                          street_name=args.street_name,
                                          storey_range=args.storey_range,
                                          floor_area_sqm=args.floor_area_sqm,
                                          model_path=model_file,
                                          data_path=data_file)
        print(f"\nPredicted Resale Price: ${prediction:,.2f}")

        # Perform trend analysis
        fig, today_trend_price = analyze_and_predict_hdb_trend(full_data_path=data_file,
                                                                town=args.town,
                                                                street_name=args.street_name,
                                                                flat_type=args.flat_type,
                                                                storey_range=args.storey_range,
                                                                floor_area_sqm=args.floor_area_sqm,
                                                                current_predicted_price=prediction)
        if fig:
            if today_trend_price:
                print(f"Today's Trend-Based Price Estimate: ${today_trend_price:,.2f}")
            plt.show()
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")