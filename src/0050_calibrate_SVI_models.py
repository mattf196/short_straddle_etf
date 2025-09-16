
# SVI Model Calibration (Parallel)
# Fits Normalized SVI volatility surface models to options data
# Processes multiple tickers in parallel with dedicated logging per ticker
#
#
# Usage:
#   python 0050_calibrate_SVI_models.py                    # incremental calibration
#   python 0050_calibrate_SVI_models.py --recalibrate      # force full recalibration
#   python 0050_calibrate_SVI_models.py --period 20250101 20250630  # date range
#
# Note: Will take a long time to run on older systems
#
# Features:
# - Multi-start optimization to avoid local minima
# - Safe parallel logging with queues
# - Skips already-processed dates unless --recalibrate
# - Outputs: calibration params, raw options data, plot data




# Parallel SVI calibration with safe logging via queues
# Each ticker gets its own log file with real-time updates

import os
import csv
import sys
import glob
import argparse
from datetime import datetime, timedelta
import multiprocessing
import logging
import time
from tqdm import tqdm

# All necessary libraries are now imported at the top level
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# =========================================
# Configuration and Target Definitions
# =========================================
LOCAL_BASE_DIR = '../ORATS'
DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d"]
TICKERS_FILE = os.path.join(LOCAL_BASE_DIR, "tickers_of_interest.csv")


# =========================================
# SVI Model and Calibration Functions
# =========================================

def normalized_svi_function(params, normalized_log_moneyness):
    a, b, rho, m, sigma = params
    return a + b * (rho * (normalized_log_moneyness - m) + np.sqrt((normalized_log_moneyness - m)**2 + sigma**2))

def multi_start_fit_normalized_svi(norm_logm, ivs, candidate_inits):
    best_obj, best_params = np.inf, None
    ivs_mean = np.mean(ivs)
    sst = np.sum((ivs - ivs_mean)**2)
    if sst == 0: return (candidate_inits[0], 0) if candidate_inits else (None, np.inf)

    for init in candidate_inits:
        bounds = [(None, None), (0, None), (-1, 1), (None, None), (0, None)]
        result = minimize(lambda p, k, v: np.sum((normalized_svi_function(p, k) - v)**2), init, args=(norm_logm, ivs), bounds=bounds)
        if result.success:
            predicted = normalized_svi_function(result.x, norm_logm)
            sse = np.sum((ivs - predicted)**2)
            r_squared = 1 - (sse / sst) if sst > 0 else 0
            if r_squared >= 0.9975: return result.x, result.fun
            if result.fun < best_obj:
                best_obj, best_params = result.fun, result.x
    return best_params, best_obj

# =========================================
# File I/O and Utility Functions
# =========================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='SVI Volatility Surface Calibration (Parallel, Dedicated Logging)')
    parser.add_argument('-r', '--recalibrate', action='store_true', help='Recalibrate all dates, including those already processed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging for skipped dates in log files')
    parser.add_argument('-p', '--period', nargs=2, metavar=('START_DATE', 'END_DATE'), help='Process only dates within the specified period (YYYYMMDD format)')
    args, unknown = parser.parse_known_args()
    return args

def is_date_in_range(date_str, start_date=None, end_date=None):
    if not start_date and not end_date: return True
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime.strptime(start_date, "%Y%m%d") if start_date else None
        end = datetime.strptime(end_date, "%Y%m%d") if end_date else None
        if start and end: return start <= date <= end
        elif start: return start <= date
        elif end: return date <= end
        return True
    except ValueError: return False

def ensure_output_dir():
    if not os.path.exists(LOCAL_BASE_DIR):
        os.makedirs(LOCAL_BASE_DIR)

def read_tickers_of_interest():
    tickers = []
    if not os.path.exists(TICKERS_FILE):
        print(f"Warning: {TICKERS_FILE} not found. Will search for any CSV in {LOCAL_BASE_DIR}")
        return tickers
    try:
        with open(TICKERS_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            tickers = [row[0].strip() for row in reader if row and row[0].strip()]
    except Exception as e:
        print(f"Error reading tickers file: {e}")
    return tickers

def get_processed_dates_from_consolidated_file(ticker, suffix):
    processed_dates = set()
    filename = os.path.join(LOCAL_BASE_DIR, f"{ticker}_{suffix}.csv")
    if not os.path.exists(filename):
        return processed_dates
    try:
        df = pd.read_csv(filename, usecols=['observation_date'], low_memory=False)
        df.dropna(subset=['observation_date'], inplace=True)
        processed_dates = set(df['observation_date'].unique())
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return processed_dates

def write_csv_output_append(data, ticker, suffix, fieldnames=None):
    if not data: return
    filename = os.path.join(LOCAL_BASE_DIR, f"{ticker}_{suffix}.csv")
    file_exists = os.path.exists(filename)
    try:
        df_to_append = pd.DataFrame(data)
        if fieldnames:
            df_to_append = df_to_append.reindex(columns=fieldnames)
        df_to_append.to_csv(filename, mode='a', header=not file_exists, index=False, float_format='%.6f')
    except Exception as e:
        # In parallel mode, it's better to log errors than print to stdout
        logging.error(f"Error writing to {filename}: {e}")

# =========================================
# Logging Setup for Parallel Processing
# =========================================

def logger_thread(q):
    """Listens for messages on the queue and writes to the correct log file."""
    while True:
        try:
            message = q.get()
            if message == 'kill':
                break
            
            ticker, line = message
            log_filename = os.path.join(LOCAL_BASE_DIR, f"{ticker}_SVI_CALIB_LOG.txt")
            with open(log_filename, 'a') as f:
                f.write(line + '\n')
                f.flush() # Explicitly flush the buffer to disk
        except Exception:
            import traceback
            traceback.print_exc()

# =========================================
# Main Worker Function
# =========================================

def process_ticker_worker(task_args):
    """
    The main worker function executed by each parallel process.
    """
    filepath, recalibrate, verbose, start_date, end_date, log_queue = task_args
    ticker = os.path.splitext(os.path.basename(filepath))[0]
    
    # Simple function to send log messages to the queue
    def log_message(line):
        log_queue.put((ticker, line))

    normalized_svi_candidates = [
        [0.1, 0.1, 0.0, 0.0, 0.1], [0.2, 0.2, 0.0, 0.0, 0.2], [0.1, 0.2, -0.5, 0.0, 0.15],
        [0.05, 0.1, 0.2, 0.0, 0.1], [0.15, 0.1, 0.5, 0.0, 0.2], [0.3, 0.05, 0.3, -0.1, 0.05],
        [0.05, 0.3, -0.3, 0.1, 0.25], [0.25, 0.15, -0.2, 0.05, 0.1], [0.2, 0.05, 0.5, 0.1, 0.15],
        [0.1, 0.3, -0.5, -0.1, 0.2], [0.0852, 0.8138, -0.5892, 0.0234, 0.002],
        [0.0808, 0.8403, -0.4214, 0.0251, 0.0046], [0.0799, 1.3156, 0.0485, 0.0244, 0.001],
        [0.0833, 0.8138, -0.5892, 0.0231, 0.0016], [0.0772, 1.3393, 0.0136, 0.0281, 0.0062],
        [0.0799, 1.3280, 0.0293, 0.0289, 0.0065], [0.0830, 0.7621, -0.5600, 0.0236, 0.0029]
    ]

    log_message(f"Starting processing for ticker: {ticker}")
    main_output_suffix = "NORM_SVI_CALIBRATIONS"
    processed_dates = set() if recalibrate else get_processed_dates_from_consolidated_file(ticker, main_output_suffix)
    log_message(f"Found {len(processed_dates)} previously processed dates.")

    try:
        df = pd.read_csv(filepath, low_memory=False)
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df.dropna(subset=['trade_date'], inplace=True)
    except Exception as e:
        log_message(f"FATAL: Could not read or parse file for {ticker}: {e}")
        return
    
    # MODIFICATION: Initialize a variable to hold the last valid 30-day IV
    last_valid_atm_30d_iv = 0.2

    for trade_date, group in df.groupby('trade_date'):
        trade_date_str = trade_date.strftime("%Y-%m-%d")

        if not is_date_in_range(trade_date_str, start_date, end_date) or trade_date_str in processed_dates:
            if verbose and not trade_date_str in processed_dates:
                 log_message(f"INFO: Skipping {trade_date_str} as it's outside the specified period.")
            continue

        rows = group.to_dict('records')
        
        # MODIFICATION: Changed expiration days from 20-40 to 7-60
        close_expiries = [
            {"days": (pd.to_datetime(r.get("expirDate"), errors='coerce') - trade_date).days, 
             "mid_iv": (float(r.get("cMidIv", "0") or 0) + float(r.get("pMidIv", "0") or 0)) / 2}
            for r in rows 
            if 7 <= (pd.to_datetime(r.get("expirDate"), errors='coerce') - trade_date).days <= 60
            and float(r.get("stkPx", 0)) > 0 and 0.98 <= float(r["strike"]) / float(r.get("stkPx", 0)) <= 1.02
        ]
        
        # MODIFICATION: Use last valid value as fallback instead of a fixed default
        if close_expiries:
            atm_30d_iv = sorted(close_expiries, key=lambda x: abs(x["days"] - 30))[0]["mid_iv"]
            last_valid_atm_30d_iv = atm_30d_iv  # Update the last known valid value
        else:
            atm_30d_iv = last_valid_atm_30d_iv # Use the last known valid value

        expirations = {}
        for row in rows:
            try:
                expir_obj = pd.to_datetime(row["expirDate"], errors='coerce')
                if pd.isna(expir_obj) or expir_obj <= trade_date: continue
                T_days = (expir_obj - trade_date).days
                T_year = T_days / 365.25
                spot_px, strike = float(row.get("stkPx", 0)), float(row["strike"])
                if spot_px <= 0: continue
                forward = spot_px * np.exp((float(row.get("iRate", 0))/100.0 - float(row.get("divRate", 0))/100.0) * T_year)
                
                option_type = "call" if strike >= forward else "put"
                mid_iv_str = row.get("cMidIv") if option_type == 'call' else row.get("pMidIv")
                mid_iv = float(mid_iv_str) if mid_iv_str else 0.0

                if mid_iv <= 0: continue
                denominator = atm_30d_iv * np.sqrt(T_year)
                if denominator <= 0: continue
                normalized_log_moneyness = np.log(strike / forward) / denominator
                if np.isnan(normalized_log_moneyness): continue
                record = {**row, "T": T_days, "T_year": T_year, "forward": forward, "strike": strike, "normalized_log_moneyness": normalized_log_moneyness, "mid_iv": mid_iv, "option_type": option_type}
                expirations.setdefault(T_year, {"data": [], "forward": forward, "spot_px": spot_px})["data"].append(record)
            except (ValueError, TypeError, ZeroDivisionError): continue
        
        all_calibrations, successful_calibrations, sub_standard_calibrations = [], 0, 0
        raw_options_data, plot_data = [], []
        
        for T_year, exp_info in sorted(expirations.items()):
            valid_data = exp_info["data"]
            if not valid_data or not any(-1.5 >= r["normalized_log_moneyness"] for r in valid_data) or not any(1.5 <= r["normalized_log_moneyness"] for r in valid_data):
                sub_standard_calibrations += 1
                continue
            
            norm_sv_multiplier, best_r2, best_params, best_data = 2.0, -np.inf, None, None
            while norm_sv_multiplier >= 1.0:
                filtered = [r for r in valid_data if -norm_sv_multiplier <= r["normalized_log_moneyness"] <= norm_sv_multiplier]
                if len(filtered) < 5:
                    norm_sv_multiplier -= 0.25
                    continue
                
                norm_logm_arr, mid_iv_arr = np.array([r["normalized_log_moneyness"] for r in filtered]), np.array([r["mid_iv"] for r in filtered])
                params, _ = multi_start_fit_normalized_svi(norm_logm_arr, mid_iv_arr, normalized_svi_candidates)
                if params is not None:
                    pred_ivs = normalized_svi_function(params, norm_logm_arr)
                    sse, sst = np.sum((mid_iv_arr - pred_ivs)**2), np.sum((mid_iv_arr - np.mean(mid_iv_arr))**2)
                    r2 = 1 - (sse / sst) if sst > 0 else 0
                    if r2 > best_r2: best_r2, best_params, best_data = r2, params, filtered
                    if r2 >= 0.99: break
                norm_sv_multiplier -= 0.25
            
            if best_params is not None:
                if best_r2 < 0.95: sub_standard_calibrations += 1
                else: successful_calibrations += 1
                
                calib_res = {"ticker": ticker, "observation_date": trade_date_str, "expiration_date": (trade_date + timedelta(days=valid_data[0]["T"])).strftime("%Y-%m-%d"), "expiration_days": valid_data[0]["T"], **dict(zip(["norm_svi_a", "norm_svi_b", "norm_svi_rho", "norm_svi_m", "norm_svi_sigma"], best_params)), "r_squared": best_r2, "spot_price": exp_info["spot_px"], "forward_price": exp_info["forward"], "atm_iv": min(valid_data, key=lambda r: abs(r["strike"] - exp_info["forward"]))["mid_iv"], "atm_30d_iv": atm_30d_iv, "min_norm_log_moneyness": min(r['normalized_log_moneyness'] for r in best_data), "max_norm_log_moneyness": max(r['normalized_log_moneyness'] for r in best_data)}
                all_calibrations.append(calib_res)
                
                # --- Populate RAW_OPTIONS and PLOT_DATA ---
                plot_data.append({**calib_res, "success": True})
                
                pred_ivs = normalized_svi_function(best_params, np.array([r["normalized_log_moneyness"] for r in best_data]))
                for r, fitted_iv in zip(best_data, pred_ivs):
                    raw_options_data.append({
                        "ticker": ticker, "observation_date": trade_date_str, "expiration_date": (trade_date + timedelta(days=r["T"])).strftime('%Y-%m-%d'),
                        "days_to_expiration": r["T"], "option_type": "C" if r["option_type"] == "call" else "P",
                        "strike": r["strike"], "spot_price": exp_info["spot_px"], "forward": r["forward"],
                        "log_moneyness": np.log(r["strike"]/r["forward"]), "normalized_log_moneyness": r["normalized_log_moneyness"],
                        "atm_30d_iv": atm_30d_iv, "implied_volatility": r["mid_iv"],
                        "norm_svi_fitted_implied_volatility": fitted_iv
                    })

            else:
                sub_standard_calibrations += 1

        total_expirations_attempted = successful_calibrations + sub_standard_calibrations
        summary_line = f"-> Ticker: {ticker:<5s} | Date: {trade_date_str}"
        if total_expirations_attempted > 0:
            pct_successful = (successful_calibrations / total_expirations_attempted) * 100
            pct_sub_standard = (sub_standard_calibrations / total_expirations_attempted) * 100
            summary_line += f" | Total: {total_expirations_attempted:<3d} | High-Q (R2>=.95): {successful_calibrations:<3d} ({pct_successful:5.1f}%) | Sub-Q (R2<.95): {sub_standard_calibrations:<3d} ({pct_sub_standard:5.1f}%)"
        else:
            summary_line += " | Status: No valid expirations found to calibrate."
        log_message(summary_line)
        
        if all_calibrations:
            write_csv_output_append(all_calibrations, ticker, "NORM_SVI_CALIBRATIONS")
        if raw_options_data:
            write_csv_output_append(raw_options_data, ticker, "NORM_SVI_RAW_OPTIONS")
        if plot_data:
            write_csv_output_append(plot_data, ticker, "NORM_SVI_PLOT_DATA")
    
    return f"Finished processing for {ticker}."

# =========================================
# Main Execution Orchestrator
# =========================================
def main():
    args = parse_arguments()
    start_date, end_date = (args.period[0], args.period[1]) if args.period else (None, None)
    if start_date or end_date:
        print(f"Processing dates from {start_date or 'beginning'} to {end_date or 'end'}")

    ensure_output_dir()
    tickers = read_tickers_of_interest()
    if not tickers:
        all_files = glob.glob(os.path.join(LOCAL_BASE_DIR, "*.csv"))
        known_outputs = ["_NORM_SVI_CALIBRATIONS.csv", "_NORM_SVI_RAW_OPTIONS.csv", "_NORM_SVI_PLOT_DATA.csv", "_SVI_CALIB_LOG.txt"]
        
        processed_tickers = set()
        for file in all_files:
            basename = os.path.basename(file)
            for suffix in known_outputs:
                if basename.endswith(suffix):
                    processed_tickers.add(basename.replace(suffix, ''))
                    break
        
        input_tickers = set()
        for file in all_files:
            basename = os.path.basename(file)
            if basename.endswith(".csv") and not any(basename.endswith(s) for s in known_outputs) and "tickers_of_interest.csv" not in basename:
                input_tickers.add(os.path.splitext(basename)[0])
        
        tickers = sorted(list(input_tickers))


    if not tickers:
        print("No tickers or CSV files found to process. Exiting.")
        return

    print(f"\nFound {len(tickers)} tickers to process: {', '.join(tickers)}")
    ticker_files = [os.path.join(LOCAL_BASE_DIR, f"{t}.csv") for t in tickers]
    tasks_to_run = [f for f in ticker_files if os.path.exists(f)]

    if not tasks_to_run:
        print("No valid input files found for the specified tickers. Exiting.")
        return

    # Set up the logging queue and listener process
    log_queue = multiprocessing.Manager().Queue()
    listener = multiprocessing.Process(target=logger_thread, args=(log_queue,))
    listener.start()

    tasks = [(f, args.recalibrate, args.verbose, start_date, end_date, log_queue) for f in tasks_to_run]
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"\nStarting parallel processing with {num_cores} cores for {len(tasks)} tickers...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_ticker_worker, tasks), total=len(tasks), desc="Calibrating Tickers"):
            pass

    # Signal the logger to terminate
    log_queue.put('kill')
    listener.join()

    print("\n" + "="*80)
    print("All tickers processed successfully! Check log files for details.")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
