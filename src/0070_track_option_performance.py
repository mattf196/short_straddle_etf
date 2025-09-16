# Options Strategy Backtesting Engine
# High-fidelity simulation using business-day calendar from actual market data
# Supports multiple option strategies with delta hedging
#
#
# Features:
# - Business day expiration calculation (N trading days, not calendar days)
# - Earnings avoidance (configurable skip windows)
# - Stock split avoidance
# - Parallel processing across tickers
# - Detailed logging with optional debug mode
#
# Usage:
#   python 0070_track_option_performance.py                      # standard run
#   python 0070_track_option_performance.py --enable-trade-log  # enable detailed logs
#   python 0070_track_option_performance.py --debug-ticker SPY  # debug specific ticker

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import glob
import argparse
from scipy.interpolate import interp1d
import bisect
import multiprocessing
from tqdm import tqdm
import logging
import csv

# =========================================
# Configuration
# =========================================
LOCAL_BASE_DIR = os.path.join('..', 'ORATS')
TRACKOPT_DIR = os.path.join(LOCAL_BASE_DIR, 'trackopt')
DATE_FORMAT_INPUT = "%Y%m%d"
DATE_FORMAT_OUTPUT = "%Y-%m-%d"

CALIBRATION_FILE_PATTERN = "{}_NORM_SVI_CALIBRATIONS.csv"
DIVIDEND_FILE_PATTERN = "{}_DIVRATE.csv"
OUTPUT_FILENAME_DETAILED_PATTERN = "{}_tracking_results_detailed_{}.csv"
OUTPUT_FILENAME_AGGREGATED_PATTERN = "{}_aggregated_returns_by_date_{}.csv"
LOG_FILENAME_PATTERN = "trade_construction_log_{}_{}.log"
DEBUG_LOG_FILENAME_PATTERN = "debug_log_{}.log"

TICKER_LIST_FILE = os.path.join(LOCAL_BASE_DIR, "tickers_of_interest.csv")
CMT_RATES_FILE = os.path.join(LOCAL_BASE_DIR, "cmt_yields_history.csv")
CALIBRATION_DIR = LOCAL_BASE_DIR
TRADE_CONFIG_FILE = os.path.join(LOCAL_BASE_DIR, 'trades_of_interest.csv')

EXPECTED_TRADE_CONFIG_COLUMNS = ['trade_name', 'start_date', 'end_date', 'target_tenor', 'target_delta1', 'target_delta2', 'earnings_skip_days']

FRED_CMT_SERIES_MATURITY_MAP = {
    "DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12, "DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5, "DGS7": 7,
    "DGS10": 10, "DGS20": 20, "DGS30": 30
}

# =========================================
# Command-Line Argument Parsing
# =========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-Ticker Hyper-Detailed Options Strategy Tracking Simulation.')
    parser.add_argument('--enable-trade-log', action='store_true',
                        help='Enable the creation of the standard trade_construction_log files for each trade.')
    parser.add_argument('--debug-ticker', type=str, default=None,
                        help='Specify a single ticker (e.g., SPY) to generate a detailed debug log.')
    parser.add_argument('--debug-start-date', type=str, default=None,
                        help='Start date (YYYYMMDD) for the debug logging window. Requires --debug-ticker.')
    parser.add_argument('--debug-end-date', type=str, default=None,
                        help='End date (YYYYMMDD) for the debug logging window. Requires --debug-ticker.')
    args = parser.parse_args()
    return args

# =========================================
# Directory Setup
# =========================================
def setup_trackopt_directories(trade_configs):
    """Create trackopt directory structure with subdirectories for each trade."""
    # Create main trackopt directory
    os.makedirs(TRACKOPT_DIR, exist_ok=True)
    
    # Create subdirectories for each unique trade name
    trade_names = set(config['trade_name'] for config in trade_configs)
    for trade_name in trade_names:
        trade_dir = os.path.join(TRACKOPT_DIR, trade_name)
        os.makedirs(trade_dir, exist_ok=True)
    
    print(f"Created trackopt directory structure with {len(trade_names)} trade subdirectories")
    return trade_names

def get_output_paths(trade_name, ticker, is_summary=False):
    """Get appropriate output paths for trade-specific or summary files."""
    if is_summary:
        # Summary files go in the main trackopt directory
        detailed_path = os.path.join(TRACKOPT_DIR, OUTPUT_FILENAME_DETAILED_PATTERN.format(trade_name, ticker))
        aggregated_path = os.path.join(TRACKOPT_DIR, OUTPUT_FILENAME_AGGREGATED_PATTERN.format(trade_name, ticker))
    else:
        # Trade-specific files go in the trade subdirectory
        trade_dir = os.path.join(TRACKOPT_DIR, trade_name)
        detailed_path = os.path.join(trade_dir, OUTPUT_FILENAME_DETAILED_PATTERN.format(trade_name, ticker))
        aggregated_path = os.path.join(trade_dir, OUTPUT_FILENAME_AGGREGATED_PATTERN.format(trade_name, ticker))
    
    return detailed_path, aggregated_path

def get_log_paths(trade_name, ticker, debug_ticker=None):
    """Get appropriate log paths for trade-specific files."""
    trade_dir = os.path.join(TRACKOPT_DIR, trade_name)
    
    # Standard trade log goes in trade subdirectory
    log_path = os.path.join(trade_dir, LOG_FILENAME_PATTERN.format(ticker, trade_name))
    
    # Debug log goes in main trackopt directory (spans all trades)
    debug_log_path = None
    if debug_ticker:
        debug_log_path = os.path.join(TRACKOPT_DIR, DEBUG_LOG_FILENAME_PATTERN.format(ticker))
    
    return log_path, debug_log_path

# =========================================
# Financial Models & Utilities
# =========================================
def black_scholes_price(option_type, S, K, T, r, q, sigma):
    if T <= 1e-9: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
    if sigma <= 0 or K <= 0 or pd.isna(sigma): return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.nan

def black_scholes_delta(option_type, S, K, T, r, q, sigma):
    if sigma <= 0 or K <= 0 or pd.isna(sigma): return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'Call':
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'Put':
        return -np.exp(-q * T) * norm.cdf(-d1)
    return np.nan

def black_scholes_gamma(S, K, T, r, q, sigma):
    if T <= 1e-9 or sigma <= 0 or K <= 0 or pd.isna(sigma): return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def get_interest_rate(trade_date, T_year, cmt_df_loaded):
    if cmt_df_loaded is None or cmt_df_loaded.empty: return 0.0
    try:
        trade_date_ts = pd.Timestamp(trade_date)
        daily_yields_series = cmt_df_loaded.asof(trade_date_ts)
        if daily_yields_series is None or daily_yields_series.isnull().all():
            if cmt_df_loaded.empty: return 0.0
            daily_yields_series = cmt_df_loaded.loc[cmt_df_loaded.first_valid_index()]
            if daily_yields_series is None or daily_yields_series.isnull().all(): return 0.0
        maturities, yields = [], []
        for col, val in daily_yields_series.items():
            if col in FRED_CMT_SERIES_MATURITY_MAP and pd.notna(val):
                maturities.append(FRED_CMT_SERIES_MATURITY_MAP[col])
                yields.append(val / 100.0)
        maturities_arr, yields_arr = np.array(maturities), np.array(yields)
        if maturities_arr.size > 0:
            combined_data = pd.DataFrame({'maturity': maturities_arr, 'yield': yields_arr}).drop_duplicates(subset=['maturity']).sort_values('maturity')
            maturities_filtered, yields_filtered = combined_data['maturity'].values, combined_data['yield'].values
        else:
            maturities_filtered, yields_filtered = np.array([]), np.array([])
        if len(maturities_filtered) < 2: return np.mean(yields_filtered) if yields_filtered.size > 0 else 0.0
        if not np.all(np.diff(maturities_filtered) > 0): return np.mean(yields_filtered) if yields_filtered.size > 0 else 0.0
        interp_func = interp1d(maturities_filtered, yields_filtered, kind='linear', bounds_error=False, fill_value="extrapolate")
        return max(0.0, float(interp_func(T_year)))
    except Exception: return 0.0

def get_dividend_rate(obs_date, expir_date, div_rate_db):
    if div_rate_db is None or div_rate_db.empty: return 0.0
    try:
        av_dates = div_rate_db.index.get_level_values('observation_date').unique()
        if av_dates.empty: return 0.0
        lookup_date = av_dates.asof(pd.Timestamp(obs_date))
        if pd.isna(lookup_date):
            lookup_date = av_dates[0]
            if pd.isna(lookup_date): return 0.0
        term_struc = div_rate_db.loc[lookup_date].reset_index().sort_values('T')
        if term_struc.empty: return 0.0
        term_struc_filtered = term_struc.dropna(subset=['T', 'final_div_rate']).drop_duplicates(subset=['T']).sort_values('T')
        if term_struc_filtered.empty or len(term_struc_filtered) < 2: return np.mean(term_struc_filtered['final_div_rate']) if not term_struc_filtered.empty else 0.0
        if not np.all(np.diff(term_struc_filtered['T'].values) > 0): return np.mean(term_struc_filtered['final_div_rate']) if not term_struc_filtered.empty else 0.0
        interp_func = interp1d(term_struc_filtered['T'], term_struc_filtered['final_div_rate'], kind='linear', bounds_error=False, fill_value="extrapolate")
        target_T = (expir_date - obs_date).days / 365.25
        rate = float(interp_func(target_T)) if target_T >= 0 else 0.0
        return rate if pd.notna(rate) else 0.0
    except Exception: return 0.0

def normalized_svi_function(params, normalized_log_moneyness):
    a, b, rho, m, sigma = params
    return a + b * (rho * (normalized_log_moneyness - m) + np.sqrt((normalized_log_moneyness - m)**2 + sigma**2))

def calculate_forward_price(spot_price, rate, dividend_rate, t_years):
    return spot_price * np.exp((rate - dividend_rate) * t_years)

def get_interpolated_iv_surface(target_date_calib, ttm_days, norm_log_m):
    calib_by_day = {int(r['expiration_days']): {'params': (r['norm_svi_a'], r['norm_svi_b'], r['norm_svi_rho'], r['norm_svi_m'], r['norm_svi_sigma']), 'min_nlmk': r['min_norm_log_moneyness'], 'max_nlmk': r['max_norm_log_moneyness']} for _, r in target_date_calib.iterrows() if not r[['norm_svi_a', 'min_norm_log_moneyness']].isnull().any()}
    if not calib_by_day: return np.nan, np.nan, np.nan
    available_days = sorted(calib_by_day.keys())
    if not available_days: return np.nan, np.nan, np.nan
    day1 = max([d for d in available_days if d <= ttm_days], default=min(available_days))
    day2 = min([d for d in available_days if d >= ttm_days], default=max(available_days))
    data1, data2 = calib_by_day[day1], calib_by_day[day2]
    iv1 = normalized_svi_function(data1['params'], np.clip(norm_log_m, data1['min_nlmk'], data1['max_nlmk']))
    iv2 = normalized_svi_function(data2['params'], np.clip(norm_log_m, data2['min_nlmk'], data2['max_nlmk']))
    if pd.isna(iv1) and pd.isna(iv2): return np.nan, day1, day2
    iv1 = iv2 if pd.isna(iv1) else iv1
    iv2 = iv1 if pd.isna(iv2) else iv2
    t = (ttm_days - day1) / (day2 - day1) if day2 > day1 else 0.0
    iv = iv1 + t * (iv2 - iv1)
    return (iv, day1, day2) if (pd.notna(iv) and iv > 0) else (np.nan, day1, day2)

def find_strike_for_delta(target_delta, option_type, ttm_days, incept_date, calib_df, cmt_df, div_df):
    ttm_years = ttm_days / 365.25
    exp_date = incept_date + timedelta(days=ttm_days)
    spot = calib_df['spot_price'].median()
    scaling_vol = calib_df['atm_30d_iv'].median()
    if pd.isna(spot) or pd.isna(scaling_vol) or ttm_years <= 0: return None
    irate = get_interest_rate(incept_date, ttm_years, cmt_df)
    divyld = get_dividend_rate(incept_date, exp_date, div_df)
    forward = calculate_forward_price(spot, irate, divyld, ttm_years)
    def error_function(strike):
        if strike <= 0: return 1.0
        log_m = np.log(strike / forward) if forward > 0 else 0
        norm_log_m = log_m / (scaling_vol * np.sqrt(ttm_years)) if ttm_years > 0 and scaling_vol > 0 else 0
        iv, _, _ = get_interpolated_iv_surface(calib_df, ttm_days, norm_log_m)
        if pd.isna(iv): return 1.0
        calculated_delta = black_scholes_delta(option_type, spot, strike, ttm_years, irate, divyld, iv)
        if pd.isna(calculated_delta): return 1.0
        return calculated_delta - target_delta
    try:
        strike_min, strike_max = spot * 0.4, spot * 2.0
        val_min, val_max = error_function(strike_min), error_function(strike_max)
        if np.sign(val_min) == np.sign(val_max): return None
        strike = brentq(error_function, strike_min, strike_max, xtol=1e-4, rtol=1e-4)
        return round(strike * 2) / 2
    except (ValueError, RuntimeError): return None

def get_daily_market_state(eval_date, ttm_days, strike_price, exp_date, all_calib_data, sorted_calib_dates, cmt_df, div_df):
    eval_date_str = eval_date.strftime(DATE_FORMAT_INPUT)
    ttm_years = ttm_days / 365.25
    
    if eval_date_str in all_calib_data:
        calib_df = all_calib_data[eval_date_str]
        spot = calib_df['spot_price'].median()
        scaling_vol = calib_df['atm_30d_iv'].median()
        
        if pd.isna(spot) or pd.isna(scaling_vol): return None
        
        irate = get_interest_rate(eval_date, ttm_years, cmt_df) if ttm_days > 0 else 0.0
        divyld = get_dividend_rate(eval_date, exp_date, div_df) if ttm_days > 0 else 0.0

        if ttm_days <= 0:
            return {'spot': spot, 'ivol': 0.0, 'irate': irate, 'divyld': divyld}

        forward = calculate_forward_price(spot, irate, divyld, ttm_years)
        norm_log_m = np.log(strike_price / forward) / (scaling_vol * np.sqrt(ttm_years)) if forward > 0 and scaling_vol > 0 and ttm_years > 0 else 0
        ivol, _, _ = get_interpolated_iv_surface(calib_df, ttm_days, norm_log_m)
        return {'spot': spot, 'ivol': ivol, 'irate': irate, 'divyld': divyld}
    else:
        return None

def get_daily_spot_price(obs_date, all_calib_data):
    obs_date_str = obs_date.strftime(DATE_FORMAT_INPUT)
    if obs_date_str in all_calib_data:
        return all_calib_data[obs_date_str]['spot_price'].iloc[0]
    return None

# =========================================
# Earnings Date Handling Functions
# =========================================
def load_earnings_dates(ticker):
    """
    Load earnings dates for a specific ticker from its EARNINGS_DATES.csv file.
    Returns a set of datetime.date objects representing earnings announcement dates.
    """
    earnings_dates = set()
    earnings_file = os.path.join(LOCAL_BASE_DIR, f"{ticker}_EARNINGS_DATES.csv")
    
    if os.path.exists(earnings_file):
        try:
            with open(earnings_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        date_str = row[0].strip()
                        try:
                            date_obj = datetime.strptime(date_str, "%Y%m%d").date()
                            earnings_dates.add(date_obj)
                        except ValueError:
                            print(f"Warning: Invalid date format in {earnings_file}: {date_str}")
        except Exception as e:
            print(f"Error reading earnings dates file for {ticker}: {e}")
    
    return earnings_dates

def is_earnings_skip_day(current_date, earnings_dates, earnings_skip_days, all_obs_dates):
    """
    Check if the current_date falls within the earnings skip window for any earnings date.
    Uses BUSINESS DAYS (from all_obs_dates list) rather than calendar days.
    
    Args:
        current_date: date object for the current trading day
        earnings_dates: set of earnings announcement dates
        earnings_skip_days: number of BUSINESS days before and after earnings to skip
        all_obs_dates: list of all business dates in chronological order
        
    Returns:
        True if this day should be skipped (P&L set to 0), False otherwise
    """
    if earnings_skip_days == 0 or not earnings_dates:
        return False
    
    try:
        current_index = all_obs_dates.index(current_date)
    except ValueError:
        # Current date not in business day list, shouldn't happen but handle gracefully
        return False
    
    for earnings_date in earnings_dates:
        try:
            earnings_index = all_obs_dates.index(earnings_date)
            business_days_from_earnings = abs(current_index - earnings_index)
            if business_days_from_earnings <= earnings_skip_days:
                return True
        except ValueError:
            # Earnings date not in business day list, fall back to finding closest business day
            # Find the closest business day to the earnings date
            closest_earnings_index = None
            min_calendar_days = float('inf')
            
            for i, obs_date in enumerate(all_obs_dates):
                calendar_days_diff = abs((obs_date - earnings_date).days)
                if calendar_days_diff < min_calendar_days:
                    min_calendar_days = calendar_days_diff
                    closest_earnings_index = i
                    
                # If we find an exact match or very close match, use it
                if calendar_days_diff <= 3:  # Within 3 calendar days is close enough
                    break
            
            if closest_earnings_index is not None:
                business_days_from_earnings = abs(current_index - closest_earnings_index)
                if business_days_from_earnings <= earnings_skip_days:
                    return True
    
    return False

# =========================================
# Stock Split Handling Functions
# =========================================
def load_split_dates(ticker):
    """
    Load split dates for a specific ticker from its SPLITS.csv file.
    Returns a set of datetime.date objects representing stock split dates.
    """
    split_dates = set()
    splits_file = os.path.join(LOCAL_BASE_DIR, f"{ticker}_SPLITS.csv")
    
    if os.path.exists(splits_file):
        try:
            with open(splits_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = row['split_date'].strip()
                    try:
                        date_obj = datetime.strptime(date_str, "%Y%m%d").date()
                        split_dates.add(date_obj)
                    except ValueError:
                        print(f"Warning: Invalid split date format in {splits_file}: {date_str}")
        except Exception as e:
            print(f"Error reading splits file for {ticker}: {e}")
    
    return split_dates

def trade_spans_split(inception_date, expiration_date, split_dates):
    """
    Check if a trade (from inception to expiration) spans across any split date.
    
    Args:
        inception_date: date object for trade inception
        expiration_date: date object for trade expiration  
        split_dates: set of split dates
        
    Returns:
        True if the trade spans a split (and should be avoided), False otherwise
    """
    if not split_dates:
        return False
    
    for split_date in split_dates:
        # If split occurs between inception and expiration (exclusive of inception, inclusive of expiration)
        if inception_date < split_date <= expiration_date:
            return True
    
    return False

def process_single_position(args_tuple):
    (ticker, inception_date, expiration_date, trade_legs, trade_config_info,
     cmt_df, div_df, all_calib_data, sorted_calib_dates, all_obs_dates,
     logger, debug_start_date, debug_end_date) = args_tuple

    if not trade_legs: return pd.DataFrame()

    enable_logging_for_this_position = False
    if logger:
        if debug_start_date and debug_end_date:
            if (inception_date <= debug_end_date) and (expiration_date >= debug_start_date):
                enable_logging_for_this_position = True
        else:
            enable_logging_for_this_position = True

    results = []
    
    # Load earnings dates and skip configuration
    earnings_skip_days = trade_config_info.get('earnings_skip_days', 0)
    earnings_dates = load_earnings_dates(ticker) if earnings_skip_days > 0 else set()
    
    if enable_logging_for_this_position and earnings_skip_days > 0:
        if logger: logger.info(f"Earnings skip enabled: {earnings_skip_days} days around earnings. Found {len(earnings_dates)} earnings dates for {ticker}.")
    
    try:
        start_index = all_obs_dates.index(inception_date)
        expiration_index = all_obs_dates.index(expiration_date)
    except ValueError:
        if logger: logger.error(f"Inception {inception_date} or expiration {expiration_date} not in business day list.")
        return pd.DataFrame()

    # --- Inception Day Processing ---
    ttm_cal_days_inception = (expiration_date - inception_date).days
    ttm_biz_days_inception = expiration_index - start_index
    
    initial_market_state_ref = get_daily_market_state(inception_date, ttm_cal_days_inception, trade_legs[0]['strike'], expiration_date, all_calib_data, sorted_calib_dates, cmt_df, div_df)
    if initial_market_state_ref is None: return pd.DataFrame()

    S, r, q, inception_spot = initial_market_state_ref['spot'], initial_market_state_ref['irate'], initial_market_state_ref['divyld'], initial_market_state_ref['spot']
    initial_total_value, initial_net_delta = 0.0, 0.0
    
    if enable_logging_for_this_position:
        logger.info(f"\n{'='*30}\nNEW POSITION: Inception {inception_date.strftime(DATE_FORMAT_OUTPUT)}, Expiration {expiration_date.strftime(DATE_FORMAT_OUTPUT)}, TTM_CAL: {ttm_cal_days_inception}, TTM_BIZ: {ttm_biz_days_inception}")
        logger.info(f"Initial Market: Spot={S:.2f}, Rate={r:.4f}, DivYield={q:.4f}")

    for i, leg in enumerate(trade_legs):
        iv_leg = get_daily_market_state(inception_date, ttm_cal_days_inception, leg['strike'], expiration_date, all_calib_data, sorted_calib_dates, cmt_df, div_df)['ivol']
        price_leg = black_scholes_price(leg['option_type'], S, leg['strike'], ttm_cal_days_inception / 365.25, r, q, iv_leg)
        delta_leg = black_scholes_delta(leg['option_type'], S, leg['strike'], ttm_cal_days_inception / 365.25, r, q, iv_leg)
        initial_total_value += price_leg * leg['position']
        initial_net_delta += delta_leg * leg['position']
        if enable_logging_for_this_position: logger.info(f"  Leg {i+1} ({leg['position']:+d} {leg['option_type']}): K={leg['strike']:.2f}, IV={iv_leg:.4f}, Price={price_leg:.4f}, Delta={delta_leg:.4f}")
    
    if enable_logging_for_this_position: logger.info(f"INCEPTION TOTALS: Position Value={initial_total_value:.4f}, Net Delta={initial_net_delta:.4f}")

    results.append({
        'ticker': ticker, 'obsdt': inception_date.strftime(DATE_FORMAT_OUTPUT), 'inception_date': inception_date.strftime(DATE_FORMAT_OUTPUT),
        'expiration_date': expiration_date.strftime(DATE_FORMAT_OUTPUT), 'trade_name': trade_config_info['trade_name'],
        'target_ttm_biz_days': trade_config_info['target_tenor'], 'target_delta1': trade_config_info['target_delta1'], 'target_delta2': trade_config_info['target_delta2'],
        'ttm_cal_days': ttm_cal_days_inception, 'ttm_biz_days': ttm_biz_days_inception, 'spot': S,
        'irate': r, 'divyld': q, 'net_delta': initial_net_delta, 'net_gamma': 0.0, 'optchg': 0.0, 'hdgchg': 0.0, 'totuhchg': 0.0, 'totdhchg': 0.0,
        'totuhret': 0.0, 'totdhret': 0.0, 'undret': 0.0, 'scaled_optchg': 0.0, 'scaled_hdgchg': 0.0, 'scaled_totuhchg': 0.0, 'scaled_totdhchg': 0.0,
        'current_total_value': initial_total_value, 'prev_day_spot': S, 'prev_day_net_delta': initial_net_delta, 'inception_spot': inception_spot
    })
    
    # --- Daily Evolution Loop (Business Day Clock) ---
    prev_day_net_delta, prev_day_spot, prev_total_value = initial_net_delta, S, initial_total_value
    cumulative_totuhchg, cumulative_totdhchg = 0.0, 0.0

    for i in range(start_index + 1, expiration_index + 1):
        current_date = all_obs_dates[i]
        
        log_this_day = False
        if enable_logging_for_this_position:
            if debug_start_date and debug_end_date:
                if debug_start_date <= current_date <= debug_end_date: log_this_day = True
            else: log_this_day = True
        
        if log_this_day: logger.info(f"\n--- Obs Date: {current_date.strftime(DATE_FORMAT_OUTPUT)} ---")

        ttm_cal_days = (expiration_date - current_date).days
        ttm_biz_days = expiration_index - i
        ttm_years = ttm_cal_days / 365.25

        S_current = get_daily_spot_price(current_date, all_calib_data)
        if S_current is None:
            if log_this_day: logger.warning(f"  No spot price for business day {current_date}. Skipping day.")
            continue

        all_legs_valid = True; temp_leg_calcs = []
        for leg_idx, leg in enumerate(trade_legs):
            market_state = get_daily_market_state(current_date, ttm_cal_days, leg['strike'], expiration_date, all_calib_data, sorted_calib_dates, cmt_df, div_df)
            if market_state is None or pd.isna(market_state.get('ivol')):
                all_legs_valid = False
                break
            
            iv_current = market_state['ivol']
            r_current = market_state['irate']
            q_current = market_state['divyld']

            price = black_scholes_price(leg['option_type'], S_current, leg['strike'], ttm_years, r_current, q_current, iv_current)
            delta = black_scholes_delta(leg['option_type'], S_current, leg['strike'], ttm_years, r_current, q_current, iv_current) if ttm_cal_days > 0 else 0.0
            gamma = black_scholes_gamma(S_current, leg['strike'], ttm_years, r_current, q_current, iv_current) if ttm_cal_days > 0 else 0.0
            temp_leg_calcs.append({'price': price, 'delta': delta, 'gamma': gamma, 'iv': iv_current})

        if not all_legs_valid:
            if log_this_day: logger.warning(f"  Could not get market state for a leg on {current_date}. Skipping day.")
            continue
        
        if log_this_day: logger.info(f"  Market: Spot={S_current:.2f}, TTM_CAL={ttm_cal_days}, TTM_BIZ={ttm_biz_days}, Rate={r_current:.4f}, DivYield={q_current:.4f}")

        current_total_value, current_net_delta, current_net_gamma = 0.0, 0.0, 0.0
        for leg_idx, leg in enumerate(trade_legs):
            calc = temp_leg_calcs[leg_idx]
            current_total_value += calc['price'] * leg['position']
            current_net_delta += calc['delta'] * leg['position']
            current_net_gamma += calc['gamma'] * leg['position']
        
        optchg = current_total_value - prev_total_value
        hdgchg = -prev_day_net_delta * (S_current - prev_day_spot)
        totuhchg, totdhchg = optchg, optchg + hdgchg
        totuhret = totuhchg / initial_total_value if initial_total_value != 0 else np.nan
        totdhret = totdhchg / initial_total_value if initial_total_value != 0 else np.nan
        cumulative_totuhchg += totuhchg; cumulative_totdhchg += totdhchg
        scaled_totuhchg = totuhchg / inception_spot if inception_spot != 0 else np.nan
        scaled_totdhchg = totdhchg / inception_spot if inception_spot != 0 else np.nan

        # Check for earnings skip day and zero P&L if needed (using business days)
        is_earnings_day = is_earnings_skip_day(current_date, earnings_dates, earnings_skip_days, all_obs_dates)
        if is_earnings_day:
            if log_this_day: logger.info(f"  EARNINGS SKIP DAY: Zeroing P&L for {current_date} (within {earnings_skip_days} business days of earnings)")
            optchg, hdgchg, totuhchg, totdhchg = 0.0, 0.0, 0.0, 0.0
            totuhret, totdhret = 0.0, 0.0
            scaled_totuhchg, scaled_totdhchg = 0.0, 0.0
            # Note: cumulative totals are NOT updated on earnings skip days

        if log_this_day:
            if ttm_cal_days <= 0:
                logger.info("  Expiration Day Settlement:")
                for leg_idx, (leg, calc) in enumerate(zip(trade_legs, temp_leg_calcs)): logger.info(f"    Leg {leg_idx+1} ({leg['position']:+d} {leg['option_type']}): K={leg['strike']:.2f}, Settlement Value={calc['price']:.4f}")
            else:
                logger.info("  Daily Leg Details:")
                for leg_idx, (leg, calc) in enumerate(zip(trade_legs, temp_leg_calcs)): logger.info(f"    Leg {leg_idx+1} ({leg['position']:+d} {leg['option_type']}): K={leg['strike']:.2f}, IV={calc['iv']:.4f}, Price={calc['price']:.4f}, Delta={calc['delta']:.4f}, Gamma={calc['gamma']:.4f}")
            delta_contribs = [f"({leg['position']:+d}*{calc['delta']:.4f})" for leg, calc in zip(trade_legs, temp_leg_calcs)]; gamma_contribs = [f"({leg['position']:+d}*{calc['gamma']:.4f})" for leg, calc in zip(trade_legs, temp_leg_calcs)]
            logger.info(f"  GREEK VERIFICATION:"); logger.info(f"    Net Delta: {current_net_delta:.4f} = {' + '.join(delta_contribs)}"); logger.info(f"    Net Gamma: {current_net_gamma:.4f} = {' + '.join(gamma_contribs)}")
            logger.info(f"  P&L: OptChg={optchg:+.4f}, HdgChg={hdgchg:+.4f} (from prev_delta={prev_day_net_delta:.4f} * spot_chg={S_current-prev_day_spot:+.2f})")
            logger.info(f"  P&L: Total Unhedged={totuhchg:+.4f}, Total D-Hedged={totdhchg:+.4f}")
            logger.info(f"  P&L Cumulative: Unhedged={cumulative_totuhchg:+.4f}, D-Hedged={cumulative_totdhchg:+.4f}")
            logger.info(f"  SCALED P&L VERIFICATION (by Inception Spot={inception_spot:.2f}):")
            logger.info(f"    Scaled Unhedged: {scaled_totuhchg:.6f} = {totuhchg:+.4f} / {inception_spot:.2f}")
            logger.info(f"    Scaled D-Hedged: {scaled_totdhchg:.6f} = {totdhchg:+.4f} / {inception_spot:.2f}")

        results.append({
            'ticker': ticker, 'obsdt': current_date.strftime(DATE_FORMAT_OUTPUT), 'inception_date': inception_date.strftime(DATE_FORMAT_OUTPUT),
            'expiration_date': expiration_date.strftime(DATE_FORMAT_OUTPUT), 'trade_name': trade_config_info['trade_name'],
            'target_ttm_biz_days': trade_config_info['target_tenor'], 'target_delta1': trade_config_info['target_delta1'], 'target_delta2': trade_config_info['target_delta2'],
            'ttm_cal_days': ttm_cal_days, 'ttm_biz_days': ttm_biz_days, 'spot': S_current,
            'irate': r_current, 'divyld': q_current, 'net_delta': current_net_delta, 'net_gamma': current_net_gamma, 'optchg': optchg, 'hdgchg': hdgchg,
            'totuhchg': totuhchg, 'totdhchg': totdhchg, 'totuhret': totuhret, 'totdhret': totdhret, 'undret': (S_current - prev_day_spot) / prev_day_spot if prev_day_spot != 0 else np.nan,
            'scaled_optchg': optchg / inception_spot if inception_spot != 0 else np.nan, 'scaled_hdgchg': hdgchg / inception_spot if inception_spot != 0 else np.nan,
            'scaled_totuhchg': scaled_totuhchg, 'scaled_totdhchg': scaled_totdhchg,
            'current_total_value': current_total_value, 'prev_day_spot': prev_day_spot, 'prev_day_net_delta': prev_day_net_delta, 'inception_spot': inception_spot
        })
        prev_total_value, prev_day_net_delta, prev_day_spot = current_total_value, current_net_delta, S_current

    return pd.DataFrame(results)

def _process_ticker_wrapper(args):
    ticker, trade_configs, cmt_df, debug_ticker_arg, debug_start, debug_end, enable_log = args
    return process_ticker(ticker, trade_configs, cmt_df, debug_ticker_arg, debug_start, debug_end, enable_log)

def process_ticker(ticker, trade_configs_for_this_ticker, cmt_df, debug_ticker, debug_start_date, debug_end_date, enable_trade_log):
    debug_logger = None
    if ticker == debug_ticker:
        # Debug log goes in main trackopt directory (spans all trades)
        _, debug_log_path = get_log_paths('dummy', ticker, debug_ticker)
        debug_logger = logging.getLogger(f'debug_log_{ticker}'); debug_logger.setLevel(logging.INFO)
        if not debug_logger.handlers:
            debug_file_handler = logging.FileHandler(debug_log_path, mode='w'); debug_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s')); debug_logger.addHandler(debug_file_handler)
    
    div_file = os.path.join(CALIBRATION_DIR, DIVIDEND_FILE_PATTERN.format(ticker));
    try:
        div_df = pd.read_csv(div_file, parse_dates=['observation_date', 'expiration_date']); div_df['T'] = (div_df['expiration_date'] - div_df['observation_date']).dt.days / 365.25; div_df = div_df.set_index(['observation_date', 'T']).sort_index()
    except Exception: div_df = pd.DataFrame()
    calibration_file = os.path.join(CALIBRATION_DIR, CALIBRATION_FILE_PATTERN.format(ticker))
    if not os.path.exists(calibration_file): return
    full_calib_df = pd.read_csv(calibration_file, parse_dates=['observation_date', 'expiration_date'])
    all_calib_data = {obs_date_str: group_df for obs_date_str, group_df in full_calib_df.groupby(full_calib_df['observation_date'].dt.strftime(DATE_FORMAT_INPUT))}
    sorted_calib_dates = sorted(all_calib_data.keys())
    if not all_calib_data: return
    all_obs_dates = sorted([datetime.strptime(d, DATE_FORMAT_INPUT).date() for d in sorted_calib_dates])
    
    # Load split dates once per ticker (used for all trade configs)
    split_dates = load_split_dates(ticker)
    
    for trade_config in trade_configs_for_this_ticker:
        output_filename_prefix = trade_config['trade_name']
        
        # Debug: Track split-related blocking for this trade config
        split_blocks_count = 0
        total_trade_attempts = 0
        
        ticker_logger = None
        if enable_trade_log:
            # Trade log goes in trade-specific subdirectory
            log_file_path, _ = get_log_paths(output_filename_prefix, ticker)
            ticker_logger = logging.getLogger(f'ticker_log_{ticker}_{output_filename_prefix}'); ticker_logger.setLevel(logging.INFO)
            if ticker_logger.hasHandlers(): ticker_logger.handlers.clear()
            file_handler = logging.FileHandler(log_file_path, mode='w'); file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s')); ticker_logger.addHandler(file_handler)
            ticker_logger.info(f"Starting processing for ticker: {ticker}, trade: {output_filename_prefix}")
        
        if debug_logger: debug_logger.info(f"\n{'='*50}\nSTARTING TRADE CONFIG: {output_filename_prefix}\n{'='*50}")

        start_date_filter, end_date_filter = trade_config['start_date'], trade_config['end_date']
        target_tenor, target_delta1, target_delta2 = trade_config['target_tenor'], trade_config['target_delta1'], trade_config['target_delta2']
        
        all_inception_data = []; active_position = None
        for i, obs_date in enumerate(tqdm(all_obs_dates, total=len(all_obs_dates), desc=f"Finding inceptions for {ticker} ({output_filename_prefix})")):
            if not (start_date_filter <= obs_date <= end_date_filter): continue
            
            should_roll = (active_position is None) or (obs_date >= active_position['expiration_date'])

            if should_roll:
                incept_date = obs_date
                total_trade_attempts += 1
                
                # --- Business Day Expiration Calculation & Validity Check ---
                expiration_index = i + target_tenor
                if expiration_index >= len(all_obs_dates):
                    if ticker_logger: ticker_logger.info(f"Stopping trade initiation on {incept_date}: tenor of {target_tenor} business days extends beyond available data.")
                    break 
                
                expiration_date = all_obs_dates[expiration_index]
                
                # --- Split Avoidance Check ---
                spans_split = trade_spans_split(incept_date, expiration_date, split_dates)
                if spans_split:
                    split_blocks_count += 1
                    print(f"SPLIT DEBUG: {ticker} BLOCKED trade {incept_date} -> {expiration_date} (spans split)")
                    if ticker_logger: ticker_logger.info(f"Skipping trade inception on {incept_date}: would span split date (expires {expiration_date})")
                    continue  # Skip this inception, continue to next observation date
                
                ttm_calendar_days = (expiration_date - incept_date).days
                current_day_calib_df = all_calib_data[incept_date.strftime(DATE_FORMAT_INPUT)]
                new_legs = []

                # --- Leg Construction Logic ---
                if output_filename_prefix.startswith('shortstraddle_longput'):
                    call_s = find_strike_for_delta(target_delta1, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df); put_s = find_strike_for_delta(-target_delta1, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df); put_l = find_strike_for_delta(-target_delta2, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    if call_s and put_s and put_l: new_legs.extend([{'option_type': 'Call', 'strike': call_s, 'position': -1}, {'option_type': 'Put', 'strike': put_s, 'position': -1}, {'option_type': 'Put', 'strike': put_l, 'position': 1}])
                elif output_filename_prefix.startswith('shortstrangle') or output_filename_prefix.startswith('shortstraddle'):
                    call_k = find_strike_for_delta(target_delta1, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df); put_k = find_strike_for_delta(-target_delta1, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    if call_k and put_k: new_legs.extend([{'option_type': 'Call', 'strike': call_k, 'position': -1}, {'option_type': 'Put', 'strike': put_k, 'position': -1}])
                elif output_filename_prefix.startswith('shortcallspread'):
                    short_k = find_strike_for_delta(target_delta1, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df); long_k = find_strike_for_delta(target_delta2, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    if short_k and long_k: new_legs.extend([{'option_type': 'Call', 'strike': short_k, 'position': -1}, {'option_type': 'Call', 'strike': long_k, 'position': 1}])
                elif output_filename_prefix.startswith('shortputspread'):
                    short_k = find_strike_for_delta(-target_delta1, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df); long_k = find_strike_for_delta(-target_delta2, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    if short_k and long_k: new_legs.extend([{'option_type': 'Put', 'strike': short_k, 'position': -1}, {'option_type': 'Put', 'strike': long_k, 'position': 1}])
                elif output_filename_prefix.startswith('shortironfly'):
                    body_call_k = find_strike_for_delta(target_delta1, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    body_put_k = find_strike_for_delta(-target_delta1, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    wing_put_k = find_strike_for_delta(-target_delta2, 'Put', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    wing_call_k = find_strike_for_delta(target_delta2, 'Call', ttm_calendar_days, incept_date, current_day_calib_df, cmt_df, div_df)
                    if body_call_k and body_put_k and wing_put_k and wing_call_k:
                        new_legs.extend([
                            {'option_type': 'Call', 'strike': body_call_k, 'position': -1},
                            {'option_type': 'Put', 'strike': body_put_k, 'position': -1},
                            {'option_type': 'Put', 'strike': wing_put_k, 'position': 1},
                            {'option_type': 'Call', 'strike': wing_call_k, 'position': 1}
                        ])

                if new_legs:
                    if ticker_logger: ticker_logger.info(f"  Successfully constructed new position on {incept_date} expiring on {expiration_date} (TTM: {ttm_calendar_days} calendar days) with legs: {new_legs}")
                    active_position = {'inception_date': incept_date, 'expiration_date': expiration_date}
                    all_inception_data.append((ticker, incept_date, expiration_date, new_legs, trade_config, cmt_df, div_df, all_calib_data, sorted_calib_dates, all_obs_dates, debug_logger, debug_start_date, debug_end_date))
                else:
                    if ticker_logger: ticker_logger.warning(f"  Failed to construct new position on {incept_date}. Strike finding may have failed.")
                    active_position = None
        
        # Debug summary for this trade config
        print(f"SPLIT DEBUG: {ticker} {output_filename_prefix} - Total attempts: {total_trade_attempts}, Split blocks: {split_blocks_count}, Valid positions: {len(all_inception_data)}")
        
        if not all_inception_data:
            if ticker_logger: ticker_logger.info(f"No valid positions were generated for {ticker} for {output_filename_prefix}."); continue

        results_list = [process_single_position(data_tuple) for data_tuple in tqdm(all_inception_data, total=len(all_inception_data), desc=f"Simulating {ticker} ({output_filename_prefix})")]
        
        # Debug: Check for empty results after position processing
        empty_results = sum(1 for df in results_list if df.empty)
        valid_results = len(results_list) - empty_results
        print(f"POSITION DEBUG: {ticker} {output_filename_prefix} - Processed positions: {len(all_inception_data)}, Valid results: {valid_results}, Empty results: {empty_results}")
        
        if not results_list or all(df.empty for df in results_list):
            if ticker_logger: ticker_logger.info(f"No valid results after simulation for {ticker} for {output_filename_prefix}."); continue
        results_df = pd.concat(results_list).dropna(subset=['spot']).reset_index(drop=True)
        if results_df.empty:
            if ticker_logger: ticker_logger.info(f"No valid results after concatenation for {ticker} for {output_filename_prefix}."); continue

        # Use new directory structure - trade-specific files go in trade subdirectory
        output_path_detailed, output_path_agg = get_output_paths(output_filename_prefix, ticker)
        results_df.to_csv(output_path_detailed, index=False, float_format='%.6f')
        if ticker_logger: ticker_logger.info(f"Saved detailed results to: {output_path_detailed}")
        
        pnl_cols = ['optchg', 'hdgchg', 'totuhchg', 'totdhchg', 'totuhret', 'totdhret', 'scaled_optchg', 'scaled_hdgchg', 'scaled_totuhchg', 'scaled_totdhchg']
        grouped = results_df.groupby('obsdt')
        agg_pnl = grouped[pnl_cols].sum(); agg_greeks = grouped[['net_delta', 'net_gamma']].sum(); pos_count = grouped.size().to_frame('pos_count'); agg_undret = grouped['undret'].mean().to_frame('undret')
        agg_df = pd.concat([agg_pnl, agg_greeks, pos_count, agg_undret], axis=1).reset_index()
        agg_df['avg_delta_per_pos'] = agg_df['net_delta'] / agg_df['pos_count']
        agg_df['avg_gamma_per_pos'] = agg_df['net_gamma'] / agg_df['pos_count']
        agg_df.to_csv(output_path_agg, index=False, float_format='%.6f')
        if ticker_logger: ticker_logger.info(f"Saved aggregated results to: {output_path_agg}")

def main():
    args = parse_arguments()
    debug_start, debug_end = None, None
    if args.debug_start_date or args.debug_end_date:
        if not (args.debug_ticker and args.debug_start_date and args.debug_end_date):
            print("FATAL: If using debug dates, you must provide --debug-ticker, --debug-start-date, AND --debug-end-date.")
            return
        try:
            debug_start = datetime.strptime(args.debug_start_date, DATE_FORMAT_INPUT).date()
            debug_end = datetime.strptime(args.debug_end_date, DATE_FORMAT_INPUT).date()
        except ValueError:
            print("FATAL: Invalid date format for debug dates. Please use YYYYMMDD.")
            return
            
    try:
        cmt_df = pd.read_csv(CMT_RATES_FILE, parse_dates=['DATE']).set_index('DATE').sort_index()
        tickers = pd.read_csv(TICKER_LIST_FILE, header=None).iloc[:, 0].tolist()
        trade_configs_df = pd.read_csv(TRADE_CONFIG_FILE)
        if not all(col in trade_configs_df.columns for col in EXPECTED_TRADE_CONFIG_COLUMNS): raise ValueError(f"Missing columns in {TRADE_CONFIG_FILE}")
        trade_configs_df['start_date'] = pd.to_datetime(trade_configs_df['start_date'], format=DATE_FORMAT_INPUT).dt.date
        trade_configs_df['end_date'] = pd.to_datetime(trade_configs_df['end_date'], format=DATE_FORMAT_INPUT).dt.date
        print(f"Loaded {len(tickers)} tickers and {len(trade_configs_df)} trade configurations.")
        
        # Set up the new trackopt directory structure
        trade_configs_list = trade_configs_df.to_dict('records')
        setup_trackopt_directories(trade_configs_list)
        
        if args.debug_ticker: print(f"** Debug mode enabled for ticker: {args.debug_ticker} **")
    except Exception as e:
        print(f"FATAL: Error during initial data loading. Error: {e}")
        return

    ticker_args = [(ticker, trade_configs_list, cmt_df, args.debug_ticker, debug_start, debug_end, args.enable_trade_log) for ticker in tickers]

    print(f"Starting parallel processing for {len(tickers)} tickers...")
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(_process_ticker_wrapper, ticker_args), total=len(ticker_args), desc="Overall Ticker Processing"))
    
    print("\nAll ticker processing complete.")

if __name__ == "__main__":
    main()
