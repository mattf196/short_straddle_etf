
# SVI Grid Interpolator (Parallel)
# Transforms SVI calibrations into standardized IV grids on fixed tenors/Z-scores
# Uses dynamic rate lookups for accurate forward pricing
#
#
# Usage:
#   python 0060_interpolate_fixed_option_tenors.py                    # incremental update
#   python 0060_interpolate_fixed_option_tenors.py --tickers SPY QQQ --reprocess  # force full reprocess
#
# Creates fixed grids:
# - Tenors: 5-365 days
# - Z-scores: -2.0 to 2.0 in 0.25 steps
# - Two-dimensional interpolation (time and moneyness)
# - Uses actual dividend/interest rates for each tenor




# Parallel processing version - generates standardized IV surfaces
# Interpolates between calibrated expirations, extrapolates beyond range
# Clamps moneyness inputs to prevent unstable extrapolation


import os
import csv
import numpy as np
import pandas as pd
import inspect
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import glob
import argparse
from scipy.interpolate import interp1d
import multiprocessing ### MODIFICATION ###: Import the multiprocessing library

# =========================================
# Configuration
# =========================================
LOCAL_BASE_DIR = '../ORATS'
INPUT_SUFFIX = "_NORM_SVI_CALIBRATIONS.csv"
OUTPUT_SUFFIX = "_NORM_CALIBRATIONS_FIXED.csv"
TICKERS_FILE = os.path.join(LOCAL_BASE_DIR, "tickers_of_interest.csv")
CMT_RATES_FILE = os.path.join(LOCAL_BASE_DIR, "cmt_yields_history.csv")

FRED_CMT_SERIES_MATURITY_MAP = {
    "DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12, "DGS1": 1,
    "DGS2": 2, "DGS3": 3, "DGS5": 5, "DGS7": 7,
    "DGS10": 10, "DGS20": 20, "DGS30": 30
}

Z_LEVELS = np.arange(-2.0, 2.25, 0.25)
MAX_DAYS = 365
MIN_DAYS = 5
MIN_R2 = 0.95
DEBUG_MODE = False

# =========================================
# Debugging Utility
# =========================================
def debug_print(message):
    if DEBUG_MODE:
        # Include process ID in debug messages for clarity in parallel execution
        pid = os.getpid()
        caller_frame = inspect.getouterframes(inspect.currentframe())[1]
        print(f"[DEBUG:PID={pid}:{os.path.basename(caller_frame.filename)}:{caller_frame.function}:{caller_frame.lineno}] {message}")


# =========================================
# Rate Lookup Functions
# =========================================
def get_dividend_rate(obs_date, expir_date, div_rate_db):
    if div_rate_db is None or div_rate_db.empty:
        debug_print("DivYield: FAIL - Dividend database is None or empty.")
        return 0.0
    try:
        available_obs_dates = div_rate_db.index.get_level_values('observation_date').unique()
        if available_obs_dates.empty: return 0.0
        lookup_obs_date = available_obs_dates.asof(obs_date)
        if pd.isna(lookup_obs_date): lookup_obs_date = available_obs_dates[0]
        term_structure = div_rate_db.loc[lookup_obs_date].reset_index().sort_values('T')
        if term_structure.empty: return 0.0
        interp_func = interp1d(term_structure['T'], term_structure['final_div_rate'], kind='linear', bounds_error=False, fill_value="extrapolate")
        target_T = (expir_date - obs_date).days / 365.25
        rate = float(interp_func(target_T)) if target_T >= 0 else 0.0
        return rate if pd.notna(rate) else 0.0
    except Exception as e:
        debug_print(f"DivYield: FAIL - An unexpected error occurred: {e}")
        return 0.0

def get_interest_rate(trade_date, T_year, cmt_df_loaded):
    if cmt_df_loaded is None or cmt_df_loaded.empty:
        debug_print("IRate: FAIL - CMT dataframe is None or empty.")
        return np.nan
    try:
        daily_yields_series = cmt_df_loaded.asof(trade_date)
        if daily_yields_series is None or daily_yields_series.isnull().all():
            first_valid_index = cmt_df_loaded.first_valid_index()
            daily_yields_series = cmt_df_loaded.loc[first_valid_index] if first_valid_index is not None else None
        if daily_yields_series is None: return np.nan
        maturities, yields = [], []
        for col_name, yield_value in daily_yields_series.items():
            if col_name in FRED_CMT_SERIES_MATURITY_MAP and not pd.isna(yield_value):
                maturities.append(FRED_CMT_SERIES_MATURITY_MAP[col_name])
                yields.append(yield_value / 100.0)
        if len(maturities) < 2: return np.nan
        interp_func = interp1d(maturities, yields, kind='linear', bounds_error=False, fill_value="extrapolate")
        return max(0.0, float(interp_func(T_year)))
    except Exception as e:
        debug_print(f"IRate: FAIL - An unexpected error occurred: {e}")
        return np.nan

# =========================================
# SVI Model and Grid Generation Functions
# =========================================
def normalized_svi_function(params, normalized_log_moneyness):
    a, b, rho, m, sigma = params
    term1 = rho * (normalized_log_moneyness - m)
    term2 = np.sqrt((normalized_log_moneyness - m)**2 + sigma**2)
    return a + b * (term1 + term2)

def log_moneyness_to_strike(log_moneyness, forward_price):
    return forward_price * np.exp(log_moneyness)

def calculate_forward_price(spot_price, rate, dividend_rate, t_years):
    return spot_price * np.exp((rate - dividend_rate) * t_years)

def calculate_iv_at_z(norm_svi_params, z, min_norm_logm, max_norm_logm):
    clamped_z = np.clip(z, min_norm_logm, max_norm_logm)
    return normalized_svi_function(norm_svi_params, clamped_z)

def linear_interpolate(x, y, x_new):
    x0, x1 = x; y0, y1 = y
    if x0 == x1 or np.isnan(y0) or np.isnan(y1): return y0 if not np.isnan(y0) else np.nan
    t = (x_new - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

### MODIFICATION: Function signature and logging updated ###
def generate_grid_for_date(ticker, obs_date_str, calib_df_for_date, cmt_historical_df, div_rate_db):
    """Processes a single observation date's data to generate the fixed grid."""
    # This function now prints a single status line per date.
    print(f"  -> Processing {ticker} for observation date: {obs_date_str}")

    calib_df_for_date = calib_df_for_date[calib_df_for_date['r_squared'] >= MIN_R2].copy()
    if calib_df_for_date.empty: return []

    observation_date = datetime.strptime(obs_date_str, '%Y-%m-%d')
    spot_price = calib_df_for_date['spot_price'].median()
    scaling_vol = calib_df_for_date['atm_30d_iv'].median()

    if pd.isna(spot_price) or pd.isna(scaling_vol) or scaling_vol <= 0:
        return []

    calib_by_day = {}
    for _, row in calib_df_for_date.iterrows():
        days = int(row['expiration_days'])
        params = (row['norm_svi_a'], row['norm_svi_b'], row['norm_svi_rho'],
                  row['norm_svi_m'], row['norm_svi_sigma'])
        bounds = {'min_nlmk': row['min_norm_log_moneyness'], 'max_nlmk': row['max_norm_log_moneyness']}
        if not any(np.isnan(p) for p in params) and not any(np.isnan(b) for b in bounds.values()):
            calib_by_day[days] = {'params': params, **bounds}

    if not calib_by_day: return []
    available_days = sorted(calib_by_day.keys())
    min_calib_day, max_calib_day = min(available_days), max(available_days)

    results = []
    for target_day in range(MIN_DAYS, MAX_DAYS + 1):
        target_t_years = target_day / 365.25
        interest_rate = get_interest_rate(observation_date, target_t_years, cmt_historical_df)
        if pd.isna(interest_rate): continue

        target_expir_date = observation_date + timedelta(days=target_day)
        dividend_rate = get_dividend_rate(observation_date, target_expir_date, div_rate_db)
        forward_price = calculate_forward_price(spot_price, interest_rate, dividend_rate, target_t_years)
        if np.isnan(forward_price): continue

        day1_data, day2_data, interpolation_type = None, None, "None"
        if target_day < min_calib_day:
            day1_data, interpolation_type = calib_by_day[min_calib_day], "NearestNeighborTimeBelow"
        elif target_day > max_calib_day:
            day1_data, interpolation_type = calib_by_day[max_calib_day], "NearestNeighborTimeAbove"
        elif target_day in available_days:
            day1_data, interpolation_type = calib_by_day[target_day], "ExactMatch"
        else:
            day1 = max(d for d in available_days if d < target_day)
            day2 = min(d for d in available_days if d > target_day)
            day1_data, day2_data = calib_by_day[day1], calib_by_day[day2]
            interpolation_type = "Interpolation"

        for z in Z_LEVELS:
            iv = np.nan
            if interpolation_type in ["NearestNeighborTimeBelow", "NearestNeighborTimeAbove", "ExactMatch"]:
                iv = calculate_iv_at_z(day1_data['params'], z, day1_data['min_nlmk'], day1_data['max_nlmk'])
            elif interpolation_type == "Interpolation":
                iv1 = calculate_iv_at_z(day1_data['params'], z, day1_data['min_nlmk'], day1_data['max_nlmk'])
                iv2 = calculate_iv_at_z(day2_data['params'], z, day2_data['min_nlmk'], day2_data['max_nlmk'])
                iv = linear_interpolate([day1, day2], [iv1, iv2], target_day)

            if np.isnan(iv) or iv <= 0: continue
            log_moneyness = z * scaling_vol * np.sqrt(target_t_years)
            strike_price = log_moneyness_to_strike(log_moneyness, forward_price)
            if np.isnan(strike_price): continue

            results.append({
                'observation_date': obs_date_str,
                'target_date': target_expir_date.strftime('%Y-%m-%d'),
                'days': target_day, 'z_score': z, 'implied_volatility': iv,
                'atm_30d_iv': scaling_vol, 'spot_price': spot_price,
                'forward_price': forward_price, 'strike_price': strike_price,
                'option_type': 'C' if strike_price >= forward_price else 'P',
                'log_moneyness': log_moneyness, 'interpolation_type': interpolation_type,
                'interest_rate': interest_rate, 'dividend_yield': dividend_rate
            })
    return results

# =========================================
# Main Orchestration
# =========================================
def get_tickers_to_process(args):
    if args.tickers: return args.tickers
    tickers = set()
    try:
        with open(TICKERS_FILE, 'r') as f:
            tickers.update(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: {TICKERS_FILE} not found. Scanning directory for inputs.")
        for f in glob.glob(os.path.join(LOCAL_BASE_DIR, f"*{INPUT_SUFFIX}")):
            tickers.add(os.path.basename(f).replace(INPUT_SUFFIX, ''))
    return sorted(list(tickers))

def get_processed_dates(ticker):
    """Reads the consolidated output file to find which dates are already done."""
    output_filename = os.path.join(LOCAL_BASE_DIR, ticker + OUTPUT_SUFFIX)
    if not os.path.exists(output_filename):
        return set()
    try:
        df = pd.read_csv(output_filename, usecols=['observation_date'])
        return set(df['observation_date'].unique())
    except (pd.errors.EmptyDataError, KeyError):
        return set()
    except Exception as e:
        print(f"Warning: Could not read processed dates from {output_filename}. Error: {e}")
        return set()

def append_to_output_file(ticker, results_df):
    """Appends the new results to the consolidated output file for the ticker."""
    if results_df.empty: return
    output_filename = os.path.join(LOCAL_BASE_DIR, ticker + OUTPUT_SUFFIX)
    file_exists = os.path.exists(output_filename)

    output_columns = ['observation_date', 'target_date', 'days', 'z_score', 'strike_price',
                      'option_type', 'implied_volatility', 'atm_30d_iv', 'spot_price',
                      'forward_price', 'log_moneyness', 'interest_rate', 'dividend_yield',
                      'interpolation_type']
    results_df = results_df.reindex(columns=output_columns)

    try:
        results_df.to_csv(output_filename, mode='a', header=not file_exists, index=False, float_format='%.6f')
        print(f"Appended {len(results_df)} new rows to {output_filename}")
    except Exception as e:
        print(f"Error writing results to {output_filename}: {e}")

def process_ticker(ticker, cmt_historical_df, args):
    """Main processing function for a single ticker."""
    # This function is now designed to be called by a multiprocessing worker
    print(f"\n--- Processing Ticker: {ticker} (PID: {os.getpid()}) ---")

    # Load consolidated SVI calibrations for the ticker
    input_filename = os.path.join(LOCAL_BASE_DIR, ticker + INPUT_SUFFIX)
    if not os.path.exists(input_filename):
        print(f"Input file not found, skipping ticker: {input_filename}")
        return

    try:
        calib_df_full = pd.read_csv(input_filename, low_memory=False, parse_dates=['observation_date'])
        calib_df_full['observation_date'] = calib_df_full['observation_date'].dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error loading {input_filename}: {e}")
        return

    # Load ticker-specific dividend database
    div_rate_db = None
    db_path = os.path.join(LOCAL_BASE_DIR, f"{ticker}_DIVRATE.csv")
    if os.path.exists(db_path):
        try:
            df_div = pd.read_csv(db_path, parse_dates=['observation_date', 'expiration_date'])
            df_div['T'] = (df_div['expiration_date'] - df_div['observation_date']).dt.days / 365.25
            div_rate_db = df_div.set_index(['observation_date', 'expiration_date']).sort_index()
            print(f"  Loaded dividend database for {ticker}")
        except Exception as e: print(f"  Warning: Could not load dividend database for {ticker}. Error: {e}")
    else: print(f"  Warning: Dividend database not found for {ticker} at {db_path}. Yield will be 0.")

    # Determine which observation dates to process
    all_obs_dates = set(calib_df_full['observation_date'].unique())
    if args.reprocess:
        dates_to_process = sorted(list(all_obs_dates))
        print(f"  --reprocess flag set. Processing all {len(dates_to_process)} observation dates found for {ticker}.")
    else:
        processed_dates = get_processed_dates(ticker)
        dates_to_process = sorted(list(all_obs_dates - processed_dates))
        print(f"  Found {len(all_obs_dates)} total dates for {ticker}. {len(processed_dates)} already processed. Processing {len(dates_to_process)} new dates.")

    if not dates_to_process:
        print(f"  No new dates to process for {ticker}.")
        return

    # Group data by observation date and process each one
    all_new_results = []
    grouped_calibs = calib_df_full.groupby('observation_date')
    for obs_date_str in dates_to_process:
        calib_df_for_date = grouped_calibs.get_group(obs_date_str)
        ### MODIFICATION: Pass ticker to the processing function ###
        daily_results = generate_grid_for_date(ticker, obs_date_str, calib_df_for_date, cmt_historical_df, div_rate_db)
        if daily_results:
            all_new_results.extend(daily_results)

    # Append all new results to the consolidated file
    if all_new_results:
        append_to_output_file(ticker, pd.DataFrame(all_new_results))
    else:
        print(f"  No new valid grid points were generated for {ticker}.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate IVs at standardized Z-scores from consolidated SVI calibrations.')
    parser.add_argument('--tickers', type=str, nargs='*', help='Optional: Specify one or more tickers to process, e.g., --tickers SPY QQQ. If not provided, defaults to reading from ../ORATS/tickers_of_interest.csv.')
    parser.add_argument('--min_r2', type=float, default=MIN_R2, help=f'Minimum R² value for filtering calibrations. Default: {MIN_R2:.2f}')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of all observation dates, even if already in the output file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help=f'Number of parallel processes to use. Default: {os.cpu_count()} (all available cores).')
    return parser.parse_args()

def main():
    global MIN_R2, DEBUG_MODE
    args = parse_arguments()
    MIN_R2, DEBUG_MODE = args.min_r2, args.debug

    print("--- SVI Fixed Grid Generation (Consolidated I/O) [PARALLELIZED] ---")
    print(f"Using minimum R² threshold: {MIN_R2:.2%}")

    # Pre-load CMT rates once
    cmt_historical_df = None
    if os.path.exists(CMT_RATES_FILE):
        try:
            cmt_historical_df = pd.read_csv(CMT_RATES_FILE, index_col=0, parse_dates=True).sort_index()
            print("  Successfully loaded CMT historical rates.")
        except Exception as e: print(f"  Warning: Could not load CMT rates file. Error: {e}")
    else: print("  Warning: CMT rates file not found. Interest rate lookups will fail.")

    tickers = get_tickers_to_process(args)
    if not tickers:
        print("No tickers found to process.")
        return

    print(f"Starting parallel processing for {len(tickers)} tickers using {args.num_workers} workers.")

    # Create a list of arguments for each task. starmap needs an iterable of tuples.
    tasks = [(ticker, cmt_historical_df, args) for ticker in tickers]

    # Use a process pool to execute the tasks in parallel
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # starmap is like map, but it unpacks the argument tuples for the function
        pool.starmap(process_ticker, tasks)

    print(f"\n==========================================")
    print(f"Completed processing for all tickers.")
    print(f"==========================================")

if __name__ == "__main__":
    # This guard is essential for multiprocessing to work correctly.
    # It prevents child processes from re-running the main script code.
    main()
