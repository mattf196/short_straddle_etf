# =============================================================================
# Fetch CMT Yield Curve Data from FRED
#
# MODIFIED VERSION: This version has been updated to save the output CSV
# with the correct string mnemonic headers (e.g., "1M", "3M", "1Y")
# required by the downstream SVI calibration script. This resolves
# previous data parsing errors.
#
#
# DESCRIPTION:
# This script downloads historical Constant Maturity Treasury (CMT) yield
# data from the Federal Reserve Economic Data (FRED) service. It fetches
# data for a predefined set of maturities, merges them into a single time
# series dataframe, and saves the result to a CSV file.
#
# Usage:
#   python 0010_fetch_cmt_yield_data.py
# =============================================================================

import os
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

# === CONFIGURATION ===========================================================
# Define the local directory to save the output file.
LOCAL_BASE_DIR = '../ORATS'
OUTPUT_FILENAME = os.path.join(LOCAL_BASE_DIR, 'cmt_yields_history.csv')

# --- START MODIFICATION ---
# This dictionary now maps the FRED Series ID to the desired, human-readable
# column name that will be used in the output CSV. This ensures downstream
# scripts can correctly identify each column.
FRED_SERIES_TO_COLUMN_NAME = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y"
}
# --- END MODIFICATION ---

# Define the date range for the data download.
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime.now()
# =============================================================================

def fetch_and_prepare_data():
    """
    Fetches, merges, and prepares the CMT data from FRED.
    """
    all_series = []
    print("Fetching CMT data from FRED...")

    # Iterate through the dictionary to fetch each series.
    # The key is the FRED series ID (e.g., "DGS1MO").
    # The value is the desired column name (e.g., "1M").
    for series_id, column_name in FRED_SERIES_TO_COLUMN_NAME.items():
        try:
            print(f"  Fetching {series_id} (as {column_name})...")
            # Fetch the data series from FRED.
            series = web.DataReader(series_id, 'fred', START_DATE, END_DATE)

            # --- START MODIFICATION ---
            # Rename the series to the desired column name (e.g., "1M").
            # This is the critical change that fixes the header issue.
            series.name = column_name
            # --- END MODIFICATION ---

            all_series.append(series)

        except Exception as e:
            print(f"  Could not fetch data for {series_id}. Error: {e}")
            continue

    if not all_series:
        print("No data was fetched. Exiting.")
        return None

    # Concatenate all fetched series into a single DataFrame.
    # The 'axis=1' argument aligns the data by date (the index).
    print("\nMerging all time series...")
    merged_df = pd.concat(all_series, axis=1)

    # Sort the DataFrame by date, just in case.
    merged_df.sort_index(inplace=True)

    # The FRED data is daily. We can fill missing values (weekends, holidays)
    # by carrying forward the last known value.
    print("Forward-filling missing values for weekends and holidays...")
    merged_df.ffill(inplace=True)

    print("Data preparation complete.")
    return merged_df

def save_data_to_csv(df):
    """
    Saves the prepared DataFrame to a CSV file.
    """
    # Ensure the output directory exists.
    if not os.path.exists(LOCAL_BASE_DIR):
        print(f"Creating output directory: {LOCAL_BASE_DIR}")
        os.makedirs(LOCAL_BASE_DIR)

    # Save the DataFrame to the specified CSV file.
    # The index is saved by default, which is what we want (the date).
    try:
        df.to_csv(OUTPUT_FILENAME)
        print(f"\nSuccessfully saved CMT data to: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Error saving data to file. Error: {e}")

# =============================================================================
# Main execution block
# =============================================================================
if __name__ == "__main__":
    # Step 1: Fetch and prepare the data.
    cmt_data = fetch_and_prepare_data()

    # Step 2: If data was successfully fetched, save it.
    if cmt_data is not None:
        save_data_to_csv(cmt_data)
    else:
        print("Process finished with no data to save.")
