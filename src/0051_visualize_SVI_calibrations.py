
# SVI Visualization Tool
# Creates PDF reports showing volatility smile fits and parameter evolution
#
#
# Usage:
#   python 0051_visualize_SVI_calibrations.py                      # all tickers, all dates
#   python 0051_visualize_SVI_calibrations.py --tickers SPY QQQ    # specific tickers
#   python 0051_visualize_SVI_calibrations.py --tickers SPY --dates 20250702  # specific date
#
# Generates multi-page PDFs with:
# - Volatility smile plots (vs strike and normalized moneyness)
# - SVI parameter evolution across term structure
# - Model fit quality metrics




"""
Normalized SVI Model Visualization and Analysis Tool
=======================================

Purpose:
--------
This script generates detailed visualizations and analysis of Normalized Stochastic Volatility
Inspired (SVI) model calibrations for option market data. It creates comprehensive
PDF reports showing the quality of normalized SVI fits and various market metrics.

--- MODIFICATION ---
- This script has been redesigned to read consolidated data files (one file per
  ticker, per type) instead of date-stamped files.
- Data loading is now performed once per ticker, and the script iterates through
  the observation dates found within the files.
- The core visualization logic remains, generating one PDF report per ticker
  per observation date.
=======================================
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
import argparse
import pandas as pd

# =========================================
# Configuration and Input Files
# =========================================
LOCAL_BASE_DIR = '../ORATS'
TICKERS_FILE = os.path.join(LOCAL_BASE_DIR, "tickers_of_interest.csv")


# =========================================
# Argument Parsing
# =========================================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate normalized SVI calibration plots from consolidated data files.')
    parser.add_argument('--tickers', type=str, nargs='*',
                        help='Optional: A list of tickers to process (e.g., --tickers SPY QQQ). If not provided, uses tickers_of_interest.csv.')
    parser.add_argument('--dates', type=str, nargs='*',
                        help='Optional: Date(s) in YYYYMMDD format to filter from the data. If none provided, processes all available dates.')
    return parser.parse_args()


# =========================================
# SVI and Financial Functions
# =========================================
def normalized_svi_function(params, normalized_log_moneyness):
    """Compute implied volatility from normalized SVI parameters and normalized log-moneyness."""
    a, b, rho, m, sigma = params
    return a + b * (rho * (normalized_log_moneyness - m) + np.sqrt((normalized_log_moneyness - m)**2 + sigma**2))

def gamma_black(F, K, sigma, T):
    """Calculate Gamma in the Black model with vectorized inputs."""
    if T <= 0 or F <= 0: return np.full_like(K, np.nan, dtype=float)
    K, sigma = np.array(K), np.array(sigma)
    gamma_values = np.full_like(K, np.nan, dtype=float)
    valid_mask = (K > 0) & (sigma > 0)
    if np.any(valid_mask):
        K_valid, sigma_valid = K[valid_mask], sigma[valid_mask]
        d1 = (np.log(F / K_valid) + 0.5 * sigma_valid**2 * T) / (sigma_valid * np.sqrt(T))
        gamma_values[valid_mask] = norm.pdf(d1) / (F * sigma_valid * np.sqrt(T))
    return gamma_values


# =========================================
# Data Loading and Utilities
# =========================================
def read_tickers_of_interest():
    """Reads the list of tickers from the configured file."""
    if not os.path.exists(TICKERS_FILE):
        print(f"Warning: Tickers file not found at {TICKERS_FILE}")
        return []
    try:
        with open(TICKERS_FILE, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers file: {e}")
        return []

def load_consolidated_files_for_ticker(ticker):
    """Loads all consolidated data files for a given ticker."""
    print(f"\nLoading consolidated data for ticker: {ticker}...")
    
    suffixes = {
        "calib": "NORM_SVI_CALIBRATIONS",
        "options": "NORM_SVI_RAW_OPTIONS",
        "plot": "NORM_SVI_PLOT_DATA"
    }
    
    data_frames = {}
    
    for key, suffix in suffixes.items():
        filename = os.path.join(LOCAL_BASE_DIR, f"{ticker}_{suffix}.csv")
        if not os.path.exists(filename):
            print(f"  - WARNING: File not found, will use empty dataframe: {filename}")
            data_frames[key] = pd.DataFrame()
            continue
        
        try:
            print(f"  - Reading {filename}...")
            df = pd.read_csv(filename, low_memory=False)
            
            # Standardize date columns on load
            for date_col in ['observation_date', 'expiration_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
            
            data_frames[key] = df
        except Exception as e:
            print(f"  - ERROR loading {filename}: {e}")
            data_frames[key] = pd.DataFrame()

    return data_frames['calib'], data_frames['options'], data_frames['plot']


# =========================================
# PDF Generation
# =========================================
def create_pdf_visualizations(ticker, date_str, options_df, plot_df, calib_df):
    """Create PDF visualizations using pre-loaded and filtered data frames."""
    print(f"\n-- Generating PDF for Ticker: {ticker}, Date: {date_str} --")

    if plot_df.empty or options_df.empty:
        print(f"  - SKIPPING: Missing or empty data for this date.")
        return

    plot_info_list = []
    for index, plot_record in plot_df.iterrows():
        exp_date = plot_record.get('expiration_date')
        if not isinstance(exp_date, str): continue
        
        exp_options = options_df[options_df['expiration_date'] == exp_date]
        if exp_options.empty: continue
        
        T, rep_forward, atm_30d_iv = plot_record.get('expiration_days'), plot_record.get('forward_price'), plot_record.get('atm_30d_iv')
        norm_lower_bound, norm_upper_bound = plot_record.get('min_norm_log_moneyness'), plot_record.get('max_norm_log_moneyness')
        
        if any(pd.isna(val) for val in [T, rep_forward, atm_30d_iv, norm_lower_bound, norm_upper_bound]): continue

        t_years = T / 365.0
        norm_moneyness_range = np.linspace(norm_lower_bound, norm_upper_bound, 100)
        log_moneyness_range = norm_moneyness_range * atm_30d_iv * np.sqrt(t_years)
        strike_range = rep_forward * np.exp(log_moneyness_range)

        norm_svi_curve = np.full_like(norm_moneyness_range, np.nan)
        params = [plot_record.get(f'norm_svi_{p}') for p in ['a', 'b', 'rho', 'm', 'sigma']]
        if not any(pd.isna(p) for p in params):
            norm_svi_curve = normalized_svi_function(params, norm_moneyness_range)

        calls, puts = exp_options[exp_options['option_type'] == 'C'], exp_options[exp_options['option_type'] == 'P']
        
        plot_info_list.append({
            "expiration": exp_date, "T": T, "strike_range": strike_range,
            "norm_moneyness_range": norm_moneyness_range, "norm_svi_curve": norm_svi_curve,
            "rep_forward": rep_forward, "norm_lower_bound": norm_lower_bound,
            "norm_upper_bound": norm_upper_bound, "r2_norm_svi": plot_record.get('r_squared', np.nan),
            "strikes_call": calls['strike'].values, "iv_call": calls['implied_volatility'].values,
            "norm_logm_call": calls['normalized_log_moneyness'].values,
            "strikes_put": puts['strike'].values, "iv_put": puts['implied_volatility'].values,
            "norm_logm_put": puts['normalized_log_moneyness'].values,
            "atm_30d_iv": atm_30d_iv
        })

    if not plot_info_list:
        print(f"  - SKIPPING: No valid plot data could be constructed for this date.")
        return
        
    plot_info_list.sort(key=lambda x: x['T'])
    
    # The PDF filename is now date-stamped, even though input is consolidated.
    pdf_filename = os.path.join(LOCAL_BASE_DIR, f"{ticker}_NORM_SVI_ANALYSIS_{datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y%m%d')}.pdf")
    with PdfPages(pdf_filename) as pdf:
        for info in plot_info_list:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'SVI Fit for {ticker} - Exp: {info["expiration"]} (T={info["T"]} days) - Obs Date: {date_str}', fontsize=14)
            r2_text = f"Fit (R²: {info['r2_norm_svi']*100:.1f}%)" if pd.notna(info['r2_norm_svi']) else "Fit (Failed)"

            ax1.scatter(info["strikes_call"], info["iv_call"], color='blue', label='Call IV', marker='o', alpha=0.7)
            ax1.scatter(info["strikes_put"], info["iv_put"], color='red', label='Put IV', marker='o', alpha=0.7)
            ax1.plot(info["strike_range"], info["norm_svi_curve"], 'k--', label=r2_text)
            ax1.axvline(x=info["rep_forward"], color='green', linestyle=':', label='Forward')
            ax1.set_xlabel("Strike"); ax1.set_ylabel("Implied Volatility"); ax1.set_title("Volatility Smile vs. Strike"); ax1.legend(); ax1.grid(True, alpha=0.3)
            
            ax1_gamma = ax1.twinx()
            t_years = info["T"] / 365.0
            gamma_values = gamma_black(info["rep_forward"], info["strike_range"], info["norm_svi_curve"], t_years)
            ax1_gamma.plot(info["strike_range"], gamma_values, color='orange', linestyle='-', alpha=0.7, label='SVI Gamma')
            ax1_gamma.set_ylabel("Gamma"); ax1_gamma.legend(loc='upper right')

            ax2.scatter(info["norm_logm_call"], info["iv_call"], color='blue', marker='o', alpha=0.7, label='_nolegend_')
            ax2.scatter(info["norm_logm_put"], info["iv_put"], color='red', marker='o', alpha=0.7, label='_nolegend_')
            ax2.plot(info["norm_moneyness_range"], info["norm_svi_curve"], 'k--', label=r2_text)
            ax2.axvline(x=0, color='green', linestyle=':', label='ATM')
            ax2.axvline(x=info["norm_lower_bound"], color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(x=info["norm_upper_bound"], color='gray', linestyle=':', alpha=0.5)
            ax2.set_xlabel("Normalized Log-Moneyness"); ax2.set_ylabel("Implied Volatility"); ax2.set_title("Volatility Smile vs. NLM"); ax2.legend(); ax2.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close()

        if not calib_df.empty:
            summary_pages(pdf, ticker, date_str, calib_df)
            
    print(f"  - Successfully wrote PDF: {os.path.basename(pdf_filename)}")

def summary_pages(pdf, ticker, date_str, calib_df):
    """Generates the summary pages for the PDF report."""
    calib_df = calib_df.sort_values('expiration_days').copy()
    
    # Parameter Evolution Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f"SVI Parameter Evolution for {ticker} on {date_str}", fontsize=16)
    params_to_plot = ['norm_svi_a', 'norm_svi_b', 'norm_svi_rho', 'norm_svi_m', 'norm_svi_sigma', 'r_squared']
    for ax, param in zip(axes.flatten(), params_to_plot):
        if param in calib_df.columns:
            ax.plot(calib_df['expiration_days'], calib_df[param], marker='o', linestyle='-')
        ax.set_title(f'{param} vs. Expiration Days'); ax.set_xlabel('Days to Expiration'); ax.set_ylabel(param); ax.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig); plt.close()

    # SVI Metrics Evolution Plot
    summary_cols = ['atm_iv_norm_svi', 'atm_slope', 'left_slope', 'right_slope']
    if all(col in calib_df.columns for col in summary_cols):
        summary_df = calib_df.dropna(subset=summary_cols + ['expiration_days']).copy()
        if not summary_df.empty:
            fig_summary, axes_summary = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
            fig_summary.suptitle(f"Evolution of Normalized SVI Metrics for {ticker} on {date_str}", fontsize=16)
            summary_df['expiration_date_dt'] = pd.to_datetime(summary_df['expiration_date'])
            summary_df = summary_df.sort_values('expiration_date_dt')
            x_axis_data, x_axis_label = summary_df['expiration_date_dt'], "Expiration Date"
            
            axes_summary[0, 0].plot(x_axis_data, summary_df['atm_iv_norm_svi'], marker='o')
            axes_summary[0, 0].set_title("ATM IV from Normalized SVI"); axes_summary[0, 0].set_ylabel("IV")
            axes_summary[0, 1].plot(x_axis_data, summary_df['atm_slope'], marker='o')
            axes_summary[0, 1].set_title("ATM Slope in Normalized Space"); axes_summary[0, 1].set_ylabel("Slope")
            axes_summary[1, 0].plot(x_axis_data, summary_df['left_slope'], marker='o')
            axes_summary[1, 0].set_title("Left Slope at Wing (Normalized Space)"); axes_summary[1, 0].set_ylabel("Slope")
            axes_summary[1, 1].plot(x_axis_data, summary_df['right_slope'], marker='o')
            axes_summary[1, 1].set_title("Right Slope at Wing (Normalized Space)"); axes_summary[1, 1].set_ylabel("Slope")
            
            for ax in axes_summary.flatten(): ax.grid(True, alpha=0.5)
            fig_summary.autofmt_xdate()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); pdf.savefig(fig_summary); plt.close(fig_summary)

# =========================================
# Main Execution Orchestrator
# =========================================
def main():
    """Main function to orchestrate the visualization generation."""
    args = parse_arguments()
    
    if args.tickers:
        tickers_to_process = [t.upper() for t in args.tickers]
        print(f"Processing user-specified tickers: {', '.join(tickers_to_process)}")
    else:
        print("No tickers specified. Reading from tickers_of_interest.csv...")
        tickers_to_process = read_tickers_of_interest()
        if not tickers_to_process:
            print("No tickers found in file. Exiting.")
            sys.exit(1)
        print(f"Found tickers: {', '.join(tickers_to_process)}")

    for ticker in tickers_to_process:
        calib_df, options_df, plot_df = load_consolidated_files_for_ticker(ticker)

        if plot_df.empty or 'observation_date' not in plot_df.columns:
            print(f"No valid plot data or 'observation_date' column for {ticker}. Skipping.")
            continue
            
        available_dates = sorted(plot_df['observation_date'].dropna().unique())

        if not available_dates:
            print(f"\nNo observation dates found in the data files for ticker {ticker}. Skipping.")
            continue
        
        dates_to_process = []
        if args.dates:
            # Convert YYYY-MM-DD from file to YYYYMMDD for comparison with arg
            arg_dates_set = set(args.dates)
            for date_in_file in available_dates:
                try:
                    if datetime.strptime(date_in_file, '%Y-%m-%d').strftime('%Y%m%d') in arg_dates_set:
                        dates_to_process.append(date_in_file)
                except (ValueError, TypeError):
                    continue
            
            if not dates_to_process:
                print(f"None of the specified dates {args.dates} were found in the data for ticker {ticker}. Skipping.")
                continue
        else:
            dates_to_process = available_dates

        print(f"\nFound {len(dates_to_process)} dates to process for ticker {ticker}.")
        for date_str in dates_to_process:
            date_options_df = options_df[options_df['observation_date'] == date_str]
            date_plot_df = plot_df[plot_df['observation_date'] == date_str]
            date_calib_df = calib_df[calib_df['observation_date'] == date_str]
            
            create_pdf_visualizations(ticker, date_str, date_options_df, date_plot_df, date_calib_df)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
