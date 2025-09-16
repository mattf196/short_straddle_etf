# Dividend Yield Inference Tool (Parallelized)
# Calculates market-implied dividend yields from options data
# Uses put-call parity and delta-neutral methods
#
#
# Usage:
#   python 0040_infer_dividend_yields.py

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm
import warnings
from matplotlib.patches import Patch
import multiprocessing
from matplotlib.backends.backend_pdf import PdfPages # Added for multi-page PDF

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# =========================================
# Configuration & Argument Parsing
# =========================================
LOCAL_BASE_DIR = '../ORATS'
TICKERS_FILE = os.path.join(LOCAL_BASE_DIR, "tickers_of_interest.csv")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Surveys and generates a historical database of implied dividend yields for multiple tickers.')
    parser.add_argument('--dates', nargs=2, metavar=('START_DATE', 'END_DATE'), help='Optional: A start and end date (YYYYMMDD) to limit the analysis period.')
    parser.add_argument('--output', type=str, help='Optional: An alternative base directory for the output files.')
    return parser.parse_args()

# =========================================
# Utility Functions
# =========================================
def read_tickers_of_interest():
    """Read the list of tickers from the configured file."""
    if not os.path.exists(TICKERS_FILE):
        print(f"Error: Tickers file not found at {TICKERS_FILE}")
        return []
    try:
        with open(TICKERS_FILE, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers file: {e}")
        return []

# =========================================
# Core Financial Calculations
# =========================================
def calculate_pcp_yield(c, p, s, k, r, t):
    try:
        if t <= 0 or s <= 0: return None
        log_arg = (c - p + k * np.exp(-r * t)) / s
        if log_arg <= 0: return None
        q = - (1 / t) * np.log(log_arg)
        return q if q >= 0 else None
    except: return None

def delta_error_function(q, s, k, t, r, v, target_delta):
    if v <= 0 or t <= 0: return np.nan
    d1 = (np.log(s / k) + (r - q + 0.5 * v**2) * t) / (v * np.sqrt(t))
    return np.exp(-q * t) * norm.cdf(d1) - target_delta

def solve_for_delta_yield(delta, s, k, t, r, v):
    try:
        q = brentq(f=delta_error_function, a=0.0, b=0.5, args=(s, k, t, r, v, delta))
        return q if q >= 0 else None
    except: return None

# =========================================
# Main Processing Logic (for a single ticker)
# =========================================
def process_single_date(df_date, observation_date):
    """Analyzes all expirations for a single observation date."""
    expirations = sorted(df_date['expirDate'].unique())
    daily_results = []
    
    for expir_str in expirations:
        try:
            expir_date = datetime.strptime(expir_str, '%m/%d/%Y')
            time_to_expiry = (expir_date - observation_date).days / 365.25
            if time_to_expiry <= 0: continue

            df_expir = df_date[df_date['expirDate'] == expir_str]
            df_expir_atm = df_expir.dropna(subset=['cValue', 'pValue', 'strike'])
            if df_expir_atm.empty: continue
            
            df_expir_atm['price_diff'] = abs(df_expir_atm['cValue'] - df_expir_atm['pValue'])
            atm_forward_strike = df_expir_atm.loc[df_expir_atm['price_diff'].idxmin()]['strike']
            
            unique_strikes = sorted(df_expir['strike'].unique())
            center_index = unique_strikes.index(atm_forward_strike)
            selected_strikes = unique_strikes[max(0, center_index - 5) : min(len(unique_strikes), center_index + 6)]
            
            pcp_yields, delta_yields = [], []
            for strike in selected_strikes:
                row = df_expir[df_expir['strike'] == strike].iloc[0]
                if pcp_yield := calculate_pcp_yield(row['cValue'], row['pValue'], row['stkPx'], strike, row['iRate'], time_to_expiry):
                    pcp_yields.append(pcp_yield)
                if row['cMidIv'] > 0 and (delta_yield := solve_for_delta_yield(row['delta'], row['stkPx'], strike, time_to_expiry, row['iRate'], row['cMidIv'])):
                    delta_yields.append(delta_yield)

            if pcp_yields or delta_yields:
                daily_results.append({
                    'observation_date': observation_date, 'expiration_date': expir_date, 'T': time_to_expiry,
                    'pcp_yield': np.median(pcp_yields) if pcp_yields else np.nan,
                    'delta_yield': np.median(delta_yields) if delta_yields else np.nan,
                })
        except Exception:
            continue
    return daily_results

def process_ticker(ticker, start_date_str, end_date_str, base_output_path):
    """Main worker function that processes a single ticker from start to finish."""
    filepath = os.path.join(LOCAL_BASE_DIR, f"{ticker}.csv")
    if not os.path.exists(filepath):
        return f"Skipped {ticker}: Data file not found."

    try:
        df_full = pd.read_csv(filepath, low_memory=False)
        df_full['trade_date_obj'] = pd.to_datetime(df_full['trade_date'], format='%m/%d/%Y', errors='coerce')
        if start_date_str and end_date_str:
            start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_date_str, format='%Y%m%d')
            df_full = df_full[(df_full['trade_date_obj'] >= start_date) & (df_full['trade_date_obj'] <= end_date)]

        df_full.dropna(subset=['trade_date_obj'], inplace=True)
        cols_to_numeric = ['strike', 'stkPx', 'cValue', 'pValue', 'cMidIv', 'delta', 'iRate']
        for col in cols_to_numeric:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
    except Exception as e:
        return f"Error pre-processing data for {ticker}: {e}"

    all_observation_dates = sorted(df_full['trade_date_obj'].unique())
    if not all_observation_dates:
        return f"Skipped {ticker}: No valid observation dates found."
        
    all_results = []
    for obs_date in all_observation_dates:
        df_single_date = df_full[df_full['trade_date_obj'] == obs_date]
        daily_results = process_single_date(df_single_date, obs_date)
        if daily_results:
            all_results.extend(daily_results)

    if all_results:
        create_outputs(ticker, all_results, base_output_path)
    else:
        return f"Completed {ticker}: No valid dividend yields could be calculated."
    
    return f"Successfully processed {ticker}."

# =========================================
# Output and Plotting
# =========================================
def create_multipage_pdf_report(ts_df, ticker, base_path):
    """Generates a single, multi-page PDF with plots for each tenor."""
    # --- MODIFICATION: Define a single PDF output path ---
    pdf_filename = os.path.join(base_path, f"{ticker}_yield_comparison.pdf")
    try:
        # --- MODIFICATION: Use PdfPages to manage the multi-page file ---
        with PdfPages(pdf_filename) as pdf:
            for tenor in [30, 60, 90]:
                fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
                
                min_yield = ts_df[[f'final_{tenor}d', f'pcp_{tenor}d', f'delta_{tenor}d']].min().min()
                max_yield = ts_df[[f'final_{tenor}d', f'pcp_{tenor}d', f'delta_{tenor}d']].max().max()
                if pd.isna(min_yield) or pd.isna(max_yield): min_yield, max_yield = 0, 0.05
                y_buffer = (max_yield - min_yield) * 0.1
                y_lims = (min_yield - y_buffer, max_yield + y_buffer)

                for date, row in ts_df.iterrows():
                    if row[f'source_{tenor}d'] == 'delta':
                        for ax in axes:
                            ax.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5), color='grey', alpha=0.2, zorder=0, linewidth=0)

                axes[0].plot(ts_df.index, ts_df[f'final_{tenor}d'], color='black', label=f'Final {tenor}-Day Yield')
                axes[1].plot(ts_df.index, ts_df[f'pcp_{tenor}d'], color='blue', label=f'PCP {tenor}-Day Median Yield')
                axes[2].plot(ts_df.index, ts_df[f'delta_{tenor}d'], color='darkgreen', label=f'Delta {tenor}-Day Median Yield')

                handles, _ = axes[0].get_legend_handles_labels()
                handles.append(Patch(facecolor='grey', alpha=0.2, label='Delta Method Dominant'))
                axes[0].legend(handles=handles, loc='upper left')
                
                for i, ax in enumerate(axes):
                    ax.set_title(f'{["Final", "PCP", "Delta-Implied"][i]} {tenor}-Day Implied Dividend Yield', fontsize=12)
                    ax.set_ylim(y_lims)
                    if i != 0: ax.legend(loc='upper left')
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))

                fig.suptitle(f'{tenor}-Day Implied Dividend Yield Time-Series for {ticker}', fontsize=16, y=0.95)
                fig.autofmt_xdate()
                
                # --- MODIFICATION: Save the current figure to the PDF object ---
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    except Exception as e:
        # Silently fail on plot errors in parallel mode
        pass

def get_source(delta_val, pcp_val):
    if pd.isna(delta_val) and pd.isna(pcp_val): return None
    if pd.isna(delta_val): return 'pcp'
    if pd.isna(pcp_val): return 'delta'
    return 'delta' if delta_val <= pcp_val else 'pcp'

def create_outputs(ticker, all_results, base_path):
    """Generates all CSV and plot outputs from the full result set."""
    db_df = pd.DataFrame(all_results)
    db_df['final_div_rate'] = db_df[['pcp_yield', 'delta_yield']].min(axis=1)
    
    # Save the CSV database file
    final_db_df = db_df[['observation_date', 'expiration_date', 'T', 'final_div_rate']].copy()
    final_db_df['observation_date'] = final_db_df['observation_date'].dt.strftime('%Y-%m-%d')
    final_db_df['expiration_date'] = final_db_df['expiration_date'].dt.strftime('%Y-%m-%d')
    db_filename = os.path.join(base_path, f"{ticker}_DIVRATE.csv")
    final_db_df.to_csv(db_filename, index=False, float_format='%.8f')

    # Prepare data for plotting
    ts_data = []
    for date, group in db_df.groupby('observation_date'):
        interp_df = group.dropna(subset=['pcp_yield', 'delta_yield', 'final_div_rate']).sort_values('T')
        if len(interp_df) < 2: continue
        
        tenors_in_years = np.array([30, 60, 90]) / 365.25
        pcp_interp = np.interp(tenors_in_years, interp_df['T'], interp_df['pcp_yield'])
        delta_interp = np.interp(tenors_in_years, interp_df['T'], interp_df['delta_yield'])
        final_interp = np.interp(tenors_in_years, interp_df['T'], interp_df['final_div_rate'])
        
        ts_data.append({'date': date,
            'final_30d': final_interp[0], 'pcp_30d': pcp_interp[0], 'delta_30d': delta_interp[0], 'source_30d': get_source(delta_interp[0], pcp_interp[0]),
            'final_60d': final_interp[1], 'pcp_60d': pcp_interp[1], 'delta_60d': delta_interp[1], 'source_60d': get_source(delta_interp[1], pcp_interp[1]),
            'final_90d': final_interp[2], 'pcp_90d': pcp_interp[2], 'delta_90d': delta_interp[2], 'source_90d': get_source(delta_interp[2], pcp_interp[2]),
        })
    
    if not ts_data: return
    ts_df = pd.DataFrame(ts_data).set_index('date').sort_index()

    # --- MODIFICATION: Call the new multi-page PDF generator ---
    create_multipage_pdf_report(ts_df, ticker, base_path)

def process_ticker_wrapper(args):
    """Unpacks arguments and calls the main processing function for a single ticker."""
    return process_ticker(*args)

# =========================================
# Main Execution
# =========================================
if __name__ == '__main__':
    args = parse_arguments()
    tickers = read_tickers_of_interest()
    if not tickers: exit()

    base_output_path = args.output if args.output else LOCAL_BASE_DIR
    os.makedirs(base_output_path, exist_ok=True)

    start_date_str = args.dates[0] if args.dates else None
    end_date_str = args.dates[1] if args.dates else None
    
    tasks = [(ticker, start_date_str, end_date_str, base_output_path) for ticker in tickers]
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"\nStarting parallel processing with {num_cores} cores for {len(tasks)} tickers...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(process_ticker_wrapper, tasks), total=len(tasks)):
            pass # Keep output clean, errors will be handled within the process

    print("\nImplied Dividend Yield Survey complete.")

