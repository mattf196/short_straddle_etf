# Trading Strategy Analysis with VIX Conditioning
# Analyzes option strategy performance with composite portfolio metrics
# Includes VIX-based exit/re-entry logic and comprehensive statistics
#
#
# Usage:
#   python 0080_combine_option_performance.py
#
# Features:
# - Equally-weighted and inverse-volatility-weighted composites
# - VIX conditioning (exit when VIX > 25, re-enter after 10 days below threshold)
# - Correlation analysis and pair plots
# - Excel-compatible CSV output with NUMBERVALUE formatting
# - Statistical summaries including tail risk metrics
import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# --- Configuration ---
BASE_DIR = '../ORATS'
TRACKOPT_DIR = os.path.join(BASE_DIR, 'trackopt')
GRAPH_OUTPUT_FILE_SUFFIX = "analysis_charts.pdf"
TRADE_CONFIG_FILE = os.path.join(BASE_DIR, 'trades_of_interest.csv')
VIX_FILE_PREFIX = "vix_time_series_"
SPY_TICKER = 'SPY'
# List of tickers/keywords to exclude from composite calculations (i.e., not included in the average)
EXCLUDE_COMPOSITE_TICKERS = ['SPY', 'QQQ', 'VIX']

def get_trade_input_dir(trade_name):
    """Get the input directory path for a specific trade."""
    return os.path.join(TRACKOPT_DIR, trade_name)

def get_trade_output_dir(trade_name):
    """Get the output directory path for trade-specific results."""
    return os.path.join(TRACKOPT_DIR, trade_name)

def get_summary_output_dir():
    """Get the output directory path for summary results spanning trades/tickers."""
    return TRACKOPT_DIR

def create_strategy_composite_files(df_unhedged: pd.DataFrame, df_hedged: pd.DataFrame, trade_prefix: str, output_dir: str):
    """
    Create composite files for the strategy that include:
    - All underlying ETF strategy returns 
    - Corresponding SPY strategy returns
    - SPY underlying
    - VIX change 
    - VIX level
    All properly ordered and NUMBERVALUE formatted.
    """
    print(f"\nCreating composite files for {trade_prefix}...")
    
    def create_composite_for_hedge_type(df: pd.DataFrame, hedge_suffix: str):
        """Create composite file for a specific hedge type (unhedged or hedged)."""
        if df.empty:
            return
        
        # Collect columns for composite file
        composite_columns = []
        
        # 1. Individual ETF strategies (exclude SPY, VIX, QQQ from composite)
        etf_strategy_cols = []
        for col in df.columns:
            if hedge_suffix in col:
                # Check if it's not one of the excluded tickers
                is_excluded = False
                for excluded_ticker in EXCLUDE_COMPOSITE_TICKERS:
                    if excluded_ticker.lower() in col.lower():
                        is_excluded = True
                        break
                if not is_excluded:
                    etf_strategy_cols.append(col)
        
        etf_strategy_cols.sort()  # Alphabetical order
        composite_columns.extend(etf_strategy_cols)
        
        # 2. SPY strategy 
        spy_strategy_cols = [col for col in df.columns if 'SPY' in col and hedge_suffix in col]
        spy_strategy_cols.sort()
        composite_columns.extend(spy_strategy_cols)
        
        # 3. SPY underlying
        if 'SPY_underlying' in df.columns:
            composite_columns.append('SPY_underlying')
        
        # 4. VIX change
        if 'VIX_chg' in df.columns:
            composite_columns.append('VIX_chg')
        
        # 5. VIX level
        if 'VIX_underlying' in df.columns:
            composite_columns.append('VIX_underlying')
        
        # Create the composite DataFrame with only the selected columns
        available_columns = [col for col in composite_columns if col in df.columns]
        
        if available_columns:
            composite_df = df[available_columns].copy()
            
            # Save the composite file
            filename = f"{trade_prefix}_composite_{hedge_suffix}_returns.csv"
            save_df_to_excel_formatted_csv(composite_df, filename, output_dir)
            print(f"  ✓ Created composite file: {filename} with {len(available_columns)} columns")
        else:
            print(f"  ⚠ No valid columns found for {hedge_suffix} composite")
    
    # Create composite files for both hedge types
    create_composite_for_hedge_type(df_unhedged, "unhedged")
    create_composite_for_hedge_type(df_hedged, "hedged")

def reorder_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns according to the specified priority:
    1. Individual ETF strategies (underlying trades of interest)
    2. SPY strategy 
    3. SPY underlying
    4. VIX change
    5. VIX level
    6. Additional composite strategies (EW, Inv_Std_Weighted, etc.)
    """
    if df.empty:
        return df
    
    # Create ordered column list
    ordered_columns = []
    remaining_columns = list(df.columns)
    
    # 1. First: Individual ETF strategies (exclude SPY, VIX, QQQ)
    etf_strategy_cols = []
    for col in df.columns:
        if ('_unhedged' in col or '_hedged' in col):
            # Check if it's not one of the excluded tickers
            is_excluded = False
            for excluded_ticker in EXCLUDE_COMPOSITE_TICKERS:
                if excluded_ticker.lower() in col.lower():
                    is_excluded = True
                    break
            if not is_excluded:
                etf_strategy_cols.append(col)
    
    # Sort ETF strategies alphabetically for consistency
    etf_strategy_cols.sort()
    ordered_columns.extend(etf_strategy_cols)
    
    # 2. Second: SPY strategy columns
    spy_strategy_cols = [col for col in df.columns if 'SPY' in col and ('_unhedged' in col or '_hedged' in col)]
    spy_strategy_cols.sort()
    ordered_columns.extend(spy_strategy_cols)
    
    # 3. Third: SPY underlying
    spy_underlying_cols = [col for col in df.columns if 'SPY_underlying' in col]
    ordered_columns.extend(spy_underlying_cols)
    
    # 4. Fourth: VIX change
    vix_chg_cols = [col for col in df.columns if 'VIX_chg' in col]
    ordered_columns.extend(vix_chg_cols)
    
    # 5. Fifth: VIX level/underlying
    vix_underlying_cols = [col for col in df.columns if 'VIX_underlying' in col]
    ordered_columns.extend(vix_underlying_cols)
    
    # 6. Sixth: Additional composite strategies (EW, Inv_Std_Weighted, etc.)
    composite_cols = []
    for col in df.columns:
        if col not in ordered_columns:
            if any(pattern in col for pattern in ['EW_Avg', 'Inv_Std_Weighted', 'Combined']):
                composite_cols.append(col)
    
    # Sort composite columns to put base composites before VIX-conditioned ones
    base_composites = [col for col in composite_cols if 'VIX_' not in col]
    vix_composites = [col for col in composite_cols if 'VIX_' in col]
    base_composites.sort()
    vix_composites.sort()
    
    ordered_columns.extend(base_composites)
    ordered_columns.extend(vix_composites)
    
    # 7. Finally: Any remaining columns
    remaining_cols = [col for col in df.columns if col not in ordered_columns]
    remaining_cols.sort()
    ordered_columns.extend(remaining_cols)
    
    # Return reordered DataFrame
    return df[ordered_columns]

def save_df_to_excel_formatted_csv(df_to_save, filename, output_dir):
    """
    Saves a DataFrame to a CSV file, encapsulating ALL numeric values
    with Excel's =NUMBERVALUE() function for Mac compatibility.
    Applies column reordering before saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    try:
        # Reorder columns first
        df_reordered = reorder_columns_for_display(df_to_save)
        
        # Create a copy to avoid modifying the original DataFrame in place
        formatted_df = df_reordered.copy()
        
        # Apply NUMBERVALUE formatting to ALL numeric values (including integers)
        formatted_df = formatted_df.map(lambda x: f'=NUMBERVALUE("{x:.6f}")' if isinstance(x, (int, float, np.number)) and pd.notna(x) else x)
        
        formatted_df.to_csv(output_path, index=True)
        print(f"Successfully saved formatted data to: {output_path}")
    except Exception as e:
        print(f"\n--- ERROR --- \nCould not save data to {output_path}. Error: {e}\n---------------")

def load_vix_data():
    """
    Loads the raw VIX time series data. Calculation is deferred until alignment.
    """
    try:
        vix_file_path = glob.glob(os.path.join(BASE_DIR, f"{VIX_FILE_PREFIX}*.csv"))[0]
        df = pd.read_csv(vix_file_path, index_col='Date', parse_dates=True)
        df.index = df.index.tz_localize(None).normalize()
        
        if 'VIX_Level' in df.columns:
            vix_raw = df[['VIX_Level']].rename(columns={'VIX_Level': 'VIX_underlying'})
            print(f"Successfully loaded VIX data from {vix_file_path}")
            return vix_raw
        else:
            print("\n--- WARNING: VIX file found, but 'VIX_Level' column is missing. ---")
            return None
    except IndexError:
         print("\n--- WARNING: VIX time series file not found. VIX analysis will be skipped. ---")
         return None
    except Exception as e:
        print(f"\n--- WARNING: Could not process VIX data. Error: {e} ---")
        return None

def load_and_prepare_data(trade_prefix: str, vix_df_raw: pd.DataFrame):
    """
    Loads all strategy data from the trade subdirectory, using the SPY calendar as the master index,
    and correctly aligns VIX data before calculating the change.
    """
    # Look for files in the trade-specific subdirectory
    trade_input_dir = get_trade_input_dir(trade_prefix)
    specific_pattern = os.path.join(trade_input_dir, f"{trade_prefix}_aggregated_returns_by_date_*.csv")
    all_files = glob.glob(specific_pattern)
    if not all_files:
        print(f"\n--- WARNING: No data files found for trade prefix '{trade_prefix}' in {trade_input_dir}. Skipping. ---\n")
        return None, None

    spy_file = next((f for f in all_files if f.endswith(f'{SPY_TICKER}.csv')), None)
    if not spy_file:
        print(f"\n--- WARNING: SPY data file not found for '{trade_prefix}' in {trade_input_dir}. Cannot establish master calendar. Skipping. ---\n")
        return None, None

    master_index = pd.read_csv(spy_file, index_col='obsdt', parse_dates=True).index
    master_index = master_index.tz_localize(None).normalize()
    
    aligned_vix_df = None
    if vix_df_raw is not None:
        aligned_vix_df = vix_df_raw.reindex(master_index, method='ffill')
        aligned_vix_df['VIX_chg'] = aligned_vix_df['VIX_underlying'].diff()

    unhedged_series_list, hedged_series_list = [], []
    for f in all_files:
        try:
            df = pd.read_csv(f, index_col='obsdt', parse_dates=True)
            df.index = df.index.tz_localize(None).normalize()
            ticker = os.path.basename(f).replace('.csv', '').split('_')[-1]
            unhedged_series = df['scaled_totuhchg'].reindex(master_index).rename(f"{ticker}_unhedged")
            hedged_series = df['scaled_totdhchg'].reindex(master_index).rename(f"{ticker}_hedged")
            unhedged_series_list.append(unhedged_series)
            hedged_series_list.append(hedged_series)
        except Exception as e:
            print(f"  - WARNING: Could not process file {f}. Error: {e}")

    df_unhedged = pd.concat(unhedged_series_list, axis=1)
    df_hedged = pd.concat(hedged_series_list, axis=1)

    if aligned_vix_df is not None:
        df_unhedged = df_unhedged.join(aligned_vix_df)
        df_hedged = df_hedged.join(aligned_vix_df)

    df_unhedged = df_unhedged.fillna(0.0).sort_index()
    df_hedged = df_hedged.fillna(0.0).sort_index()
    
    spy_underlying = pd.read_csv(spy_file, index_col='obsdt', parse_dates=True)['undret']
    spy_underlying.index = spy_underlying.index.tz_localize(None).normalize()
    df_unhedged[f'{SPY_TICKER}_underlying'] = spy_underlying.reindex(master_index).fillna(0.0)
    df_hedged[f'{SPY_TICKER}_underlying'] = spy_underlying.reindex(master_index).fillna(0.0)

    print(f"Data loading and alignment for trade '{trade_prefix}' complete.")
    return df_unhedged, df_hedged


def apply_vix_conditioning(strategy_returns: pd.Series, vix_levels: pd.Series, vix_threshold: float, exit_lag: int, reentry_wait_period: int) -> pd.Series:
    """
    Applies VIX conditioning logic to a strategy's returns.
    
    If VIX > vix_threshold, exit the strategy with an exit_lag (days) and do not reenter
    until VIX remains below vix_threshold for reentry_wait_period (business days).

    Args:
        strategy_returns (pd.Series): The daily returns of the strategy (e.g., EW_Avg).
        vix_levels (pd.Series): The daily VIX levels.
        vix_threshold (float): The VIX level above which the strategy exits.
        exit_lag (int): The number of business days to lag the exit signal.
        reentry_wait_period (int): The number of consecutive business days VIX must be
                                  below the threshold before reentering.

    Returns:
        pd.Series: The VIX-conditioned daily returns of the strategy.
    """
    if strategy_returns.empty or vix_levels.empty:
        return pd.Series(index=strategy_returns.index, data=0.0)

    # Ensure both series have the same index and are aligned
    # Use .align to handle potential index mismatches while preserving NaNs from original
    strategy_returns, vix_levels = strategy_returns.align(vix_levels, join='inner')

    conditioned_returns = pd.Series(0.0, index=strategy_returns.index)
    
    in_market = True
    days_since_vix_high = 0 # Counter for days VIX has been below threshold for reentry
    days_into_exit_lag = 0 # Counter for days after VIX > threshold, within exit_lag

    for i in range(len(strategy_returns)):
        current_date = strategy_returns.index[i]
        current_vix = vix_levels.iloc[i]

        if in_market:
            if current_vix > vix_threshold:
                # VIX crossed threshold, start the exit lag countdown
                days_into_exit_lag += 1
                if days_into_exit_lag > exit_lag:
                    # After lag, go out of market
                    in_market = False
                    days_into_exit_lag = 0 # Reset for next exit
                    days_since_vix_high = 0 # Reset for reentry
                else:
                    # Still within lag, continue in market
                    conditioned_returns.loc[current_date] = strategy_returns.iloc[i]
            else:
                # VIX is below threshold, stay in market
                conditioned_returns.loc[current_date] = strategy_returns.iloc[i]
                days_into_exit_lag = 0 # Reset if VIX drops back below threshold
        else: # Not in market (waiting for reentry)
            if current_vix <= vix_threshold:
                days_since_vix_high += 1
                if days_since_vix_high >= reentry_wait_period:
                    # Condition met, re-enter market
                    in_market = True
                    days_since_vix_high = 0 # Reset for next reentry
                    conditioned_returns.loc[current_date] = strategy_returns.iloc[i]
                else:
                    # Not enough days yet, remain out of market (returns are 0)
                    conditioned_returns.loc[current_date] = 0.0
            else:
                # VIX is still high, reset counter, remain out of market
                days_since_vix_high = 0
                conditioned_returns.loc[current_date] = 0.0
    
    return conditioned_returns.fillna(0.0) # Ensure no NaNs from alignment or initial series


def add_composite_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds equally-weighted and inverse-volatility-weighted averages to the DataFrame,
    and also VIX-conditioned versions of these composites.
    Excludes specified major ETFs/indices and their derivatives/underlyings from composites.
    """
    # Filter out columns that contain any of the excluded tickers or specific VIX/SPY underlying
    filtered_strategy_cols = []
    for col in df.columns:
        # Exclude VIX_underlying, SPY_underlying, and anything matching EXCLUDE_COMPOSITE_TICKERS
        is_excluded = False
        if 'VIX_underlying' in col or 'SPY_underlying' in col:
            is_excluded = True
        else:
            for excluded_ticker in EXCLUDE_COMPOSITE_TICKERS:
                if excluded_ticker.lower() in col.lower():
                    is_excluded = True
                    break
        
        # Only include columns that are not excluded and are individual strategy returns
        # These typically have '_unhedged' or '_hedged' suffix but are not SPY/VIX/QQQ
        if not is_excluded and ('_unhedged' in col or '_hedged' in col) and not any(excluded in col for excluded in EXCLUDE_COMPOSITE_TICKERS):
             filtered_strategy_cols.append(col)

    strategy_cols = filtered_strategy_cols
    
    if not strategy_cols:
        print("--- WARNING: No valid strategy columns found for composite calculation after filtering. Setting composites to 0.0. ---")
        df['EW_Avg'], df['Inv_Std_Weighted_Avg'] = 0.0, 0.0
        return df

    print(f"Composites will be calculated based on {len(strategy_cols)} filtered strategies: {strategy_cols}")
    df['EW_Avg'] = df[strategy_cols].mean(axis=1)
    full_sample_std = df[strategy_cols].std().replace(0, np.nan).dropna()
    if not full_sample_std.empty:
        weights = (1 / full_sample_std) / (1 / full_sample_std).sum()
        df['Inv_Std_Weighted_Avg'] = (df[weights.index] * weights).sum(axis=1)
    else:
        df['Inv_Std_Weighted_Avg'] = 0.0
    print(f"Added 'EW_Avg' and 'Inv_Std_Weighted_Avg' composites based on {len(strategy_cols)} filtered strategies.")

    # --- Add VIX-conditioned composites ---
    if 'VIX_underlying' in df.columns:
        vix_thresholds = [25]
        exit_lags = [5, 10]
        reentry_wait_period = 10 # 10 business days for reentry

        for threshold in vix_thresholds:
            for lag in exit_lags:
                for composite_type in ['EW_Avg', 'Inv_Std_Weighted_Avg']:
                    if composite_type in df.columns:
                        new_col_name = f"{composite_type}_VIX_{threshold}_ExitLag_{lag}_Reentry_{reentry_wait_period}"
                        print(f"  - Calculating {new_col_name}...")
                        df[new_col_name] = apply_vix_conditioning(
                            df[composite_type],
                            df['VIX_underlying'],
                            vix_threshold=threshold,
                            exit_lag=lag,
                            reentry_wait_period=reentry_wait_period
                        )
    else:
        print("--- WARNING: VIX_underlying data not available. Skipping VIX-conditioned composites. ---")

    return df

def perform_statistical_analysis(df: pd.DataFrame, title: str, output_dir: str):
    """
    Calculates and prints a comprehensive statistical analysis table, including
    start/end dates, moments, Sharpe ratio, percentiles, and autocorrelations.
    Also saves the statistics and correlation matrix to uniquely named CSV files.
    """
    if df.empty:
        print(f"\n--- WARNING: DataFrame for '{title}' is empty. Skipping statistical analysis. ---")
        return

    print("\n" + "="*80 + f"\n Statistical Analysis: {title}\n" + "="*80)
    
    # Reorder DataFrame columns for consistent display
    df = reorder_columns_for_display(df)
    
    # --- Sanitize title to create a valid filename ---
    sanitized_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")

    # --- Calculate Start and End Dates for each series ---
    start_dates = {}
    end_dates = {}
    for col in df.columns:
        non_zero_series = df[col].ne(0)
        if non_zero_series.any():
            start_dates[col] = non_zero_series.idxmax()
            end_dates[col] = non_zero_series.iloc[::-1].idxmax()
        else:
            start_dates[col] = pd.NaT
            end_dates[col] = pd.NaT
            
    start_date_series = pd.Series(start_dates, name="Start Date")
    end_date_series = pd.Series(end_dates, name="End Date")

    # Calculate all other metrics
    count = df.count()
    mean = df.mean()
    std = df.std()
    skew = df.skew()
    kurtosis = df.kurt()
    
    # Calculate Annualized Sharpe Ratio
    annualized_mean = mean * 252
    annualized_std = std * np.sqrt(252)
    sharpe_ratio = (annualized_mean / annualized_std).replace([np.inf, -np.inf], np.nan)
    
    # Calculate Percentiles, including 0% and 100%
    percentiles_to_calculate = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    percentiles_df = df.quantile(percentiles_to_calculate).T
    percentile_labels = [f'{int(p*100)}%' for p in percentiles_to_calculate]
    percentiles_df.columns = percentile_labels
    
    # Calculate Autocorrelations for first 3 lags
    autocorrelations = pd.DataFrame(index=df.columns)
    for lag in range(1, 4):
        autocorrelations[f'Autocorr_{lag}'] = df.apply(lambda x: x.autocorr(lag=lag), axis=0)

    # Calculate max loss (0%) and 1% loss (1%) as ratio of Annual SD
    annualized_std_series = df.std() * np.sqrt(252)
    percentiles_df['Max Loss/Annual SD'] = percentiles_df['0%'] / annualized_std_series
    percentiles_df['1% Loss/Annual SD'] = percentiles_df['1%'] / annualized_std_series
    
    # Combine all metrics into a single DataFrame, adding new date columns at the front
    combined_analysis = pd.concat([
        start_date_series,
        end_date_series,
        count.rename('Count'),
        mean.rename('Mean'),
        std.rename('Std Dev'),
        skew.rename('Skew'),
        kurtosis.rename('Kurtosis'),
        sharpe_ratio.rename('Annualized Sharpe'),
        percentiles_df,
        autocorrelations
    ], axis=1)
    
    # Format date columns for display
    combined_analysis['Start Date'] = combined_analysis['Start Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    combined_analysis['End Date'] = combined_analysis['End Date'].dt.strftime('%Y-%m-%d').fillna('N/A')

    # Reorder percentile ratio columns to place them after Sharpe ratio
    cols = list(combined_analysis.columns)
    max_loss_col = 'Max Loss/Annual SD'
    one_percent_loss_col = '1% Loss/Annual SD'
    sharpe_col = 'Annualized Sharpe'
    
    if max_loss_col in cols and one_percent_loss_col in cols and sharpe_col in cols:
        cols.remove(max_loss_col)
        cols.remove(one_percent_loss_col)
        sharpe_index = cols.index(sharpe_col)
        cols.insert(sharpe_index + 1, max_loss_col)
        cols.insert(sharpe_index + 2, one_percent_loss_col)
    
    combined_analysis = combined_analysis[cols]
    
    # Reorder columns in the combined analysis for display
    combined_analysis = reorder_columns_for_display(combined_analysis.T).T
    
    print("\n--- Summary Statistics Table ---\n", combined_analysis.to_string(float_format='{:.4f}'.format))
    # --- SAVE SUMMARY STATISTICS TABLE TO CSV ---
    stats_filename = f"{sanitized_title}_summary_statistics.csv"
    save_df_to_excel_formatted_csv(combined_analysis, stats_filename, output_dir)

    print("\n--- Daily Data Correlation Matrix ---")
    correlation_matrix = df.corr()
    
    # Reorder correlation matrix for consistent display
    correlation_matrix_ordered = reorder_columns_for_display(correlation_matrix)
    correlation_matrix_ordered = correlation_matrix_ordered.reindex(correlation_matrix_ordered.columns)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(correlation_matrix_ordered)
    print("-"*(len(title)+26) + "\n")
    
    # Use the ordered matrix for saving
    correlation_matrix = correlation_matrix_ordered
    # --- SAVE CORRELATION MATRIX TO CSV ---
    corr_filename = f"{sanitized_title}_correlation_matrix.csv"
    save_df_to_excel_formatted_csv(correlation_matrix, corr_filename, output_dir)

def generate_analysis_charts(df_unhedged, df_hedged, trade_prefix, output_path):
    """
    Generates a limited multivariate scatterplot matrix and a five-panel time series plot
    for both unhedged and hedged strategies, saving them to a single PDF.
    """
    print(f"\nGenerating analysis charts for trade '{trade_prefix}'...")
    try:
        with PdfPages(output_path) as pdf:
            print("  - Generating exploratory plots...")
            
            # --- Custom plotting functions to add zero lines ---
            def hist_with_zeroline(x, **kwargs):
                ax = plt.gca()
                sns.histplot(x, **kwargs)
                # Add vertical line unless the variable is VIX_Underlying
                if 'VIX_underlying' not in x.name:
                    ax.axvline(0, color='red', linestyle='--', linewidth=1)

            def scatter_with_zerolines(x, y, **kwargs):
                ax = plt.gca()
                sns.scatterplot(x=x, y=y, **kwargs)
                # Add vertical line unless the x-axis is VIX_Underlying
                if 'VIX_underlying' not in x.name:
                    ax.axvline(0, color='red', linestyle='--', linewidth=1)
                # Add horizontal line unless the y-axis is VIX_Underlying
                if 'VIX_underlying' not in y.name:
                    ax.axhline(0, color='red', linestyle='--', linewidth=1)

            def create_plots(df_source, hedge_type, trade_prefix, pdf_object):
                # Define the specific columns to plot
                plot_vars = [
                    'EW_Avg', 'Inv_Std_Weighted_Avg', f'{SPY_TICKER}_underlying', 
                    'VIX_underlying', 'VIX_chg'
                ]
                
                # Filter for columns that actually exist
                cols_to_plot = [col for col in plot_vars if col in df_source.columns]
                
                if len(cols_to_plot) < 2:
                    print(f"  - WARNING: Too few columns to create plots for '{hedge_type} {trade_prefix}'. Skipping.")
                    return

                # --- Page 1: Limited Pair Plot ---
                page_title = f'Exploratory Analysis: Key Composites and Market Data ({hedge_type}) - {trade_prefix}'
                analysis_df = df_source[cols_to_plot].copy()
                
                agg_rules = {col: ('last' if 'VIX_underlying' in col else 'sum') for col in analysis_df.columns}
                monthly_df = analysis_df.resample('ME').agg(agg_rules).dropna()
                
                if monthly_df.shape[0] < 20: 
                    print(f"  - WARNING: Monthly aggregated data for '{page_title}' has too few rows ({monthly_df.shape[0]}). Skipping pair plot.")
                else:
                    print(f"  - Creating limited pair plot for '{page_title}'...")
                    g = sns.PairGrid(monthly_df, height=1.8)
                    g.map_diag(hist_with_zeroline, color='steelblue')
                    g.map_offdiag(scatter_with_zerolines, alpha=0.6, s=25, edgecolor='k')
                    g.fig.suptitle(page_title, y=1.03, fontsize=14)
                    pdf_object.savefig(g.fig, bbox_inches='tight')
                    plt.close(g.fig)

                # --- Page 2: Five-Panel Bar Chart Time Series Plot ---
                print(f"  - Creating monthly time series bar chart plot for '{hedge_type} {trade_prefix}'...")
                fig, axes = plt.subplots(nrows=len(cols_to_plot), ncols=1, figsize=(12, 10), sharex=True)
                fig.suptitle(f'Monthly Time Series: Key Series ({hedge_type}) - {trade_prefix}', fontsize=16)

                # Find indices for every 6th month starting from the first month
                monthly_dates = monthly_df.index
                start_month = monthly_dates[0].month
                start_year = monthly_dates[0].year
                
                tick_indices = [0]
                current_month = start_month
                current_year = start_year
                
                for i in range(1, len(monthly_dates)):
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                        current_year += 1
                    
                    if monthly_dates[i].month == current_month:
                        tick_indices.append(i)
                        
                tick_indices_filtered = [i for i in tick_indices if (monthly_dates[i].month-1) % 6 == 0 or i==0]

                for i, col in enumerate(cols_to_plot):
                    ax = axes[i]
                    # Plotting monthly aggregated series as a bar chart
                    monthly_df[col].plot(kind='bar', ax=ax, width=0.8, color='steelblue')
                    ax.set_ylabel(col, fontsize=10)
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
                    ax.axhline(0, color='red', linestyle='--', linewidth=1)
                    
                    # Set the new ticks and labels for better readability
                    ax.set_xticks(tick_indices_filtered)
                    ax.set_xticklabels([monthly_dates[j].strftime('%Y-%m') for j in tick_indices_filtered], rotation=45, ha='right')

                plt.xlabel('Date', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf_object.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            for hedge_type, df_source in [('Unhedged', df_unhedged), ('Hedged', df_hedged)]:
                create_plots(df_source, hedge_type, trade_prefix, pdf)
        
        print(f"Successfully saved analysis charts to: {output_path}")
    except Exception as e:
        print(f"\n--- ERROR --- \nCould not generate or save analysis charts. Error: {e}\n---------------")

def generate_combined_pair_plot(df: pd.DataFrame, title: str, output_path: str):
    """
    Generates an exploratory data analysis style pair plot with histograms along the diagonal
    for a given DataFrame and saves it to a PDF.
    """
    print(f"\nGenerating combined pair plot for '{title}'...")
    try:
        with PdfPages(output_path) as pdf:
            # --- Custom plotting functions to add zero lines ---
            def hist_with_zeroline(x, **kwargs):
                ax = plt.gca()
                sns.histplot(x, **kwargs)
                # Add vertical line unless the variable is VIX_Underlying
                if 'VIX_underlying' not in x.name: # Assuming VIX_Underlying is the only one not centered at 0
                    ax.axvline(0, color='red', linestyle='--', linewidth=1)

            def scatter_with_zerolines(x, y, **kwargs):
                ax = plt.gca()
                sns.scatterplot(x=x, y=y, **kwargs)
                # Add vertical line unless the x-axis is VIX_Underlying
                if 'VIX_underlying' not in x.name:
                    ax.axvline(0, color='red', linestyle='--', linewidth=1)
                # Add horizontal line unless the y-axis is VIX_Underlying
                if 'VIX_underlying' not in y.name:
                    ax.axhline(0, color='red', linestyle='--', linewidth=1)

            if df.empty or df.shape[1] < 2:
                print(f"  - WARNING: DataFrame for combined pair plot is empty or has too few columns. Skipping plot.")
                return

            # Resample to monthly for pair plots if daily data is too dense
            agg_rules = {col: ('last' if 'VIX_underlying' in col else 'sum') for col in df.columns}
            monthly_df = df.resample('ME').agg(agg_rules).dropna()

            if monthly_df.shape[0] < 2: # Need at least two data points for a meaningful plot
                print(f"  - WARNING: Monthly aggregated data for combined pair plot has too few rows ({monthly_df.shape[0]}). Skipping plot.")
                return

            print(f"  - Creating pair plot for '{title}' with {monthly_df.shape[1]} variables.")
            g = sns.PairGrid(monthly_df, height=1.8)
            
            g.map_diag(hist_with_zeroline, color='steelblue')
            g.map_offdiag(scatter_with_zerolines, alpha=0.6, s=25, edgecolor='k')
            
            g.fig.suptitle(title, y=1.03, fontsize=14)
            pdf.savefig(g.fig, bbox_inches='tight')
            plt.close(g.fig)

        print(f"Successfully saved combined pair plot to: {output_path}")
    except Exception as e:
        print(f"\n--- ERROR --- \nCould not generate or save combined pair plot. Error: {e}\n---------------")


def main():
    """
    Main function to orchestrate the loading, analysis, and saving of data.
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    
    try:
        trade_configs_df = pd.read_csv(TRADE_CONFIG_FILE)
        trade_prefixes = trade_configs_df['trade_name'].unique().tolist()
    except Exception as e:
        print(f"FATAL: Could not load or parse {TRADE_CONFIG_FILE}. Error: {e}")
        return

    vix_df_raw = load_vix_data()

    # Lists to collect various EW_Avg composites for combined analysis
    all_unhedged_ew_avg_series = []
    all_unhedged_vix_conditioned_ew_avg_series = {}
    all_hedged_ew_avg_series = []
    all_hedged_vix_conditioned_ew_avg_series = {}

    last_vix_chg_series = None 
    last_vix_underlying_series = None


    for trade_prefix in trade_prefixes:
        print(f"\n{'='*100}\n Starting analysis for Trade: {trade_prefix}\n{'='*100}\n")
        
        df_unhedged, df_hedged = load_and_prepare_data(trade_prefix, vix_df_raw)
        if df_unhedged is None: continue

        # Ensure VIX_underlying is available before adding composites
        if 'VIX_underlying' not in df_unhedged.columns:
            print(f"--- WARNING: VIX_underlying missing for {trade_prefix}. VIX-conditioned composites might be skipped.")

        df_unhedged = add_composite_variables(df_unhedged)
        df_hedged = add_composite_variables(df_hedged)
        
        # Save trade-specific results to trade subdirectory
        trade_output_dir = get_trade_output_dir(trade_prefix)
        chart_output_path = os.path.join(trade_output_dir, f"{trade_prefix}_{GRAPH_OUTPUT_FILE_SUFFIX}")
        generate_analysis_charts(df_unhedged, df_hedged, trade_prefix, chart_output_path)

        # Save time series with proper formatting and column ordering
        save_df_to_excel_formatted_csv(df_unhedged, f"{trade_prefix}_unhedged_returns_timeseries.csv", trade_output_dir)
        save_df_to_excel_formatted_csv(df_hedged, f"{trade_prefix}_hedged_returns_timeseries.csv", trade_output_dir)
        
        # Create composite files for each strategy with all underlying returns + SPY + VIX data
        create_strategy_composite_files(df_unhedged, df_hedged, trade_prefix, trade_output_dir)

        perform_statistical_analysis(df_unhedged, f"Unhedged Returns ({trade_prefix})", trade_output_dir)
        perform_statistical_analysis(df_hedged, f"Delta-Hedged Returns ({trade_prefix})", trade_output_dir)

        # Collect UNHEDGED EW_Avg and VIX-conditioned EW_Avg for combined analysis
        if 'EW_Avg' in df_unhedged.columns:
            all_unhedged_ew_avg_series.append(df_unhedged['EW_Avg'].rename(f"{trade_prefix}_EW_Avg"))
            for col in df_unhedged.columns:
                if col.startswith('EW_Avg_VIX_'):
                    if col not in all_unhedged_vix_conditioned_ew_avg_series:
                        all_unhedged_vix_conditioned_ew_avg_series[col] = []
                    all_unhedged_vix_conditioned_ew_avg_series[col].append(df_unhedged[col].rename(f"{trade_prefix}_{col}"))
        
        # Collect HEDGED EW_Avg and VIX-conditioned EW_Avg for combined analysis
        if 'EW_Avg' in df_hedged.columns:
            all_hedged_ew_avg_series.append(df_hedged['EW_Avg'].rename(f"{trade_prefix}_EW_Avg"))
            for col in df_hedged.columns:
                if col.startswith('EW_Avg_VIX_'):
                    if col not in all_hedged_vix_conditioned_ew_avg_series:
                        all_hedged_vix_conditioned_ew_avg_series[col] = []
                    all_hedged_vix_conditioned_ew_avg_series[col].append(df_hedged[col].rename(f"{trade_prefix}_{col}"))

        # Capture VIX_chg and VIX_underlying from the last successfully loaded df_unhedged
        if 'VIX_chg' in df_unhedged.columns:
            last_vix_chg_series = df_unhedged['VIX_chg']
        if 'VIX_underlying' in df_unhedged.columns:
            last_vix_underlying_series = df_unhedged['VIX_underlying']

    # --- Combined Analysis of All UNHEDGED EW_Avg Trades and VIX Data ---
    print(f"\n{'='*100}\n Starting Combined Analysis of All UNHEDGED Composites and VIX Data\n{'='*100}\n")

    if not all_unhedged_ew_avg_series:
        print("--- WARNING: No UNHEDGED EW_Avg series collected for combined analysis. Skipping unhedged combined analysis. ---")
    else:
        # Concatenate all unhedged EW_Avg series into a single DataFrame
        combined_unhedged_df_for_analysis = pd.concat(all_unhedged_ew_avg_series, axis=1)

        # Join with VIX_chg data if available
        if last_vix_chg_series is not None:
            combined_unhedged_df_for_analysis = combined_unhedged_df_for_analysis.join(last_vix_chg_series.rename('VIX_chg'), how='left')
        else:
            print("--- WARNING: VIX_chg data not available for unhedged combined analysis. Proceeding without it. ---")
        
        # Join with VIX_underlying data if available (needed for pair plot zero lines)
        if last_vix_underlying_series is not None:
            combined_unhedged_df_for_analysis = combined_unhedged_df_for_analysis.join(last_vix_underlying_series.rename('VIX_underlying'), how='left')
        else:
            print("--- WARNING: VIX_underlying data not available for unhedged combined analysis. Pair plots might be affected. ---")

        # Add combined VIX-conditioned EW_Avg series to the combined DataFrame
        for vix_col_name, series_list in all_unhedged_vix_conditioned_ew_avg_series.items():
            combined_vix_conditional_series = pd.concat(series_list, axis=1).mean(axis=1).rename(f"Combined_Unhedged_{vix_col_name}")
            combined_unhedged_df_for_analysis = combined_unhedged_df_for_analysis.join(combined_vix_conditional_series, how='left')

        # Fill NaNs that might result from join if indices don't perfectly overlap
        combined_unhedged_df_for_analysis = combined_unhedged_df_for_analysis.fillna(0.0) 
        
        # Save combined unhedged data to CSV with a more descriptive name
        summary_output_dir = get_summary_output_dir()
        combined_unhedged_csv_path = os.path.join(summary_output_dir, "all_trades_unhedged_COMPOSITE_TIMESERIES.csv")
        try:
            # Use the standardized save function with column reordering and NUMBERVALUE formatting
            save_df_to_excel_formatted_csv(combined_unhedged_df_for_analysis, "all_trades_unhedged_COMPOSITE_TIMESERIES.csv", summary_output_dir)
        except Exception as e:
            print(f"\n--- ERROR --- \nCould not save combined UNHEDGED composites and VIX data to CSV. Error: {e}\n---------------")

        # Perform statistical analysis on the combined unhedged DataFrame
        perform_statistical_analysis(combined_unhedged_df_for_analysis, "All Unhedged Trades Composites and VIX Combined", summary_output_dir)

        # Generate pair plot for the combined unhedged DataFrame
        combined_unhedged_pair_plot_path = os.path.join(summary_output_dir, "all_unhedged_trades_composites_and_VIX_pair_plot.pdf")
        generate_combined_pair_plot(combined_unhedged_df_for_analysis, "Exploratory Analysis: All Unhedged Trades Composites and VIX", combined_unhedged_pair_plot_path)


    # --- Combined Analysis of All DELTA-HEDGED EW_Avg Trades and VIX Data ---
    print(f"\n{'='*100}\n Starting Combined Analysis of All DELTA-HEDGED Composites and VIX Data\n{'='*100}\n")

    if not all_hedged_ew_avg_series:
        print("--- WARNING: No DELTA-HEDGED EW_Avg series collected for combined analysis. Skipping delta-hedged combined analysis. ---")
    else:
        # Concatenate all hedged EW_Avg series into a single DataFrame
        combined_hedged_df_for_analysis = pd.concat(all_hedged_ew_avg_series, axis=1)

        # Join with VIX_chg data if available
        if last_vix_chg_series is not None:
            combined_hedged_df_for_analysis = combined_hedged_df_for_analysis.join(last_vix_chg_series.rename('VIX_chg'), how='left')
        else:
            print("--- WARNING: VIX_chg data not available for hedged combined analysis. Proceeding without it. ---")
        
        # Join with VIX_underlying data if available (needed for pair plot zero lines)
        if last_vix_underlying_series is not None:
            combined_hedged_df_for_analysis = combined_hedged_df_for_analysis.join(last_vix_underlying_series.rename('VIX_underlying'), how='left')
        else:
            print("--- WARNING: VIX_underlying data not available for hedged combined analysis. Pair plots might be affected. ---")

        # Add combined VIX-conditioned EW_Avg series to the combined DataFrame
        for vix_col_name, series_list in all_hedged_vix_conditioned_ew_avg_series.items():
            combined_vix_conditional_series = pd.concat(series_list, axis=1).mean(axis=1).rename(f"Combined_Hedged_{vix_col_name}")
            combined_hedged_df_for_analysis = combined_hedged_df_for_analysis.join(combined_vix_conditional_series, how='left')

        # Fill NaNs that might result from join if indices don't perfectly overlap
        combined_hedged_df_for_analysis = combined_hedged_df_for_analysis.fillna(0.0) 
        
        # Save combined hedged data to CSV with a more descriptive name
        summary_output_dir = get_summary_output_dir()
        combined_hedged_csv_path = os.path.join(summary_output_dir, "all_trades_hedged_COMPOSITE_TIMESERIES.csv")
        try:
            # Use the standardized save function with column reordering and NUMBERVALUE formatting
            save_df_to_excel_formatted_csv(combined_hedged_df_for_analysis, "all_trades_hedged_COMPOSITE_TIMESERIES.csv", summary_output_dir)
        except Exception as e:
            print(f"\n--- ERROR --- \nCould not save combined DELTA-HEDGED composites and VIX data to CSV. Error: {e}\n---------------")

        # Perform statistical analysis on the combined hedged DataFrame
        perform_statistical_analysis(combined_hedged_df_for_analysis, "All Delta-Hedged Trades Composites and VIX Combined", summary_output_dir)

        # Generate pair plot for the combined hedged DataFrame
        combined_hedged_pair_plot_path = os.path.join(summary_output_dir, "all_hedged_trades_composites_and_VIX_pair_plot.pdf")
        generate_combined_pair_plot(combined_hedged_df_for_analysis, "Exploratory Analysis: All Delta-Hedged Trades Composites and VIX", combined_hedged_pair_plot_path)


    print("\nAll analyses complete.")

if __name__ == "__main__":
    main()
