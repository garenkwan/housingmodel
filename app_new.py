import streamlit as st
import pandas as pd
import os
from predict_hdb_resale_price import predict_resale_price, analyze_and_predict_hdb_trend, get_hdb_dropdown_values
from predict import predict_price, analyze_and_predict_trend, load_model_and_artifacts, get_condo_dropdown_values

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Singapore Property Price Predictor")

# --- File Paths ---
HDB_DATA_FILE = 'data/ResaleflatpricesJan2017onwards-20260107.csv'
HDB_MODEL_FILE = os.path.join('artifacts', 'xgb_hdb_resale_model.joblib')

CONDO_DATA_FILE = 'data/PMI_Res_Transactions_20251230.csv'
CONDO_ARTIFACTS_DIR = 'artifacts'

# --- Caching Data Loading ---
@st.cache_data
def load_hdb_selection_data():
    """Loads HDB data for dropdowns, caching the result."""
    df = pd.read_csv(HDB_DATA_FILE, usecols=['town', 'street_name', 'block', 'flat_type', 'storey_range', 'floor_area_sqm'])
    # Drop duplicates to make the list of options smaller and cleaner
    df.drop_duplicates(inplace=True)
    return df

@st.cache_data
def load_condo_data():
    """Loads Condo data and dropdown values, caching the result."""
    return get_condo_dropdown_values(CONDO_ARTIFACTS_DIR, CONDO_DATA_FILE)

@st.cache_resource
def load_condo_model_artifacts():
    """Loads the condo model and details, caching the resource."""
    return load_model_and_artifacts(CONDO_ARTIFACTS_DIR)

# --- Main App ---
st.title("Singapore Property Price Prediction Engine")

hdb_tab, condo_tab = st.tabs(["HDB Resale Price Prediction", "Condo Price Prediction"])

# --- HDB Prediction Tab ---
with hdb_tab:
    st.header("Predict HDB Resale Price")

    try:
        hdb_df = load_hdb_selection_data()
        towns = sorted(hdb_df['town'].unique())
        flat_types = sorted(hdb_df['flat_type'].unique())
        storey_ranges = sorted(hdb_df['storey_range'].unique())

        col1, col2, col3 = st.columns(3)

        # --- Column 1: Town, Street Name ---
        with col1:
            town = st.selectbox("Town", options=towns, index=None, placeholder="Select a Town")

            street_options = []
            if town:
                street_options = sorted(hdb_df[hdb_df['town'] == town]['street_name'].unique())
            street_name = st.selectbox("Street Name", options=street_options, index=None, placeholder="Select a Street", disabled=not town)

        # --- Column 2: Block, Floor Area ---
        with col2:
            block_options = []
            if town and street_name:
                block_options = sorted(hdb_df[(hdb_df['town'] == town) & (hdb_df['street_name'] == street_name)]['block'].unique())
            block = st.selectbox("Block Number", options=block_options, index=None, placeholder="Select a Block", disabled=not street_name)

            area_options = []
            if town and street_name and block:
                area_options = sorted(hdb_df[(hdb_df['town'] == town) & (hdb_df['street_name'] == street_name) & (hdb_df['block'] == block)]['floor_area_sqm'].unique())
            floor_area_sqm = st.selectbox("Floor Area (sqm)", options=area_options, index=None, placeholder="Select an Area", disabled=not block)

        # --- Column 3: Flat Type, Storey Range ---
        with col3:
            flat_type = st.selectbox("Flat Type", options=flat_types, index=None, placeholder="Select a Flat Type")
            storey_range = st.selectbox("Storey Range", options=storey_ranges, index=None, placeholder="Select a Storey Range")

        # --- Prediction Button ---
        st.write("") # Spacer
        submitted = st.button("Predict and Analyze", disabled=not all([town, street_name, block, floor_area_sqm, flat_type, storey_range]))

        if submitted and all([town, street_name, block, floor_area_sqm, flat_type, storey_range]):
            with st.spinner("Running prediction and generating trend analysis..."):
                try:
                    prediction = predict_resale_price(
                        town=town,
                        flat_type=flat_type,
                        block=block,
                        street_name=str(street_name), # Ensure it's a string
                        storey_range=storey_range,
                        floor_area_sqm=float(floor_area_sqm), # Ensure it's a float
                        model_path=HDB_MODEL_FILE,
                        data_path=HDB_DATA_FILE
                    )
                    st.success(f"**Predicted Resale Price:** `${prediction:,.2f}`")

                    # Convert inputs for trend analysis
                    trend_street = str(street_name)
                    trend_area = float(floor_area_sqm)

                    fig, today_trend_price = analyze_and_predict_hdb_trend(
                        full_data_path=HDB_DATA_FILE,
                        town=town,
                        street_name=trend_street,
                        flat_type=flat_type,
                        storey_range=storey_range,
                        floor_area_sqm=trend_area,
                        current_predicted_price=prediction
                    )

                    if today_trend_price:
                        st.info(f"**Today's Trend-Based Price Estimate:** `${today_trend_price:,.2f}`")

                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Could not generate a trend analysis graph due to insufficient historical data for similar flats.")
                except (ValueError, FileNotFoundError) as e:
                    st.error(f"An error occurred: {e}")
        elif submitted:
            st.warning("Please fill in all the fields before predicting.")

    except FileNotFoundError:
        st.error(f"HDB data file not found at '{HDB_DATA_FILE}'. Please check the path.")

# --- Condo Prediction Tab ---
with condo_tab:
    st.header("Predict Private Property (Condo) Price")

    try:
        projects, floor_ranges = load_condo_data()
        model, details_df = load_condo_model_artifacts()

        with st.form("condo_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                project = st.selectbox("Project Name", options=projects)
            with col2:
                sqft = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)
                floor_range = st.selectbox("Floor Range", options=floor_ranges)
            
            submitted = st.form_submit_button("Predict and Analyze")

        if submitted:
            with st.spinner("Running prediction and generating trend analysis..."):
                try:
                    prediction = predict_price(
                        model=model,
                        project_details_df=details_df,
                        project=project,
                        sqft=sqft,
                        floorRange=floor_range
                    )
                    st.success(f"**Predicted Price:** `${prediction:,.2f}`")

                    fig, today_trend_price = analyze_and_predict_trend(
                        full_data_path=CONDO_DATA_FILE,
                        project=project,
                        floorRange=floor_range,
                        sqft=sqft,
                        current_predicted_price=prediction
                    )
                    if today_trend_price:
                        st.info(f"**Today's Trend-Based Price Estimate:** `${today_trend_price:,.2f}`")

                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Could not generate a trend analysis graph due to insufficient historical data.")

                except (ValueError, FileNotFoundError) as e:
                    st.error(f"An error occurred: {e}")

    except FileNotFoundError:
        st.error(f"A required data or model file was not found. Please check the paths in '{CONDO_ARTIFACTS_DIR}' and '{CONDO_DATA_FILE}'.")