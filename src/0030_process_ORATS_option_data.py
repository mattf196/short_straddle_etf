# ORATS ZIP File Processor
# Extracts ticker-specific data from ZIP files in parallel
# Avoids BrokenPipeError by writing temp .part files instead of returning big objects
#
#
# Usage:
#   python 0030_process_ORATS_option_data.py                    # process all new files
#   python 0030_process_ORATS_option_data.py --d 20250101 20250630  # specific date range
#
# How it works:
# 1. Reads tickers_of_interest.csv for target tickers
# 2. Checks existing {TICKER}.csv files to skip already-processed dates
# 3. Processes ZIP files in parallel, writing temp .part files
# 4. Consolidates .part files into final {TICKER}.csv files
#
# Note: Uses temp files to avoid multiprocessing data transfer issues





# Parallel processing to extract data from ORATS ZIP files
# Writes temp files to avoid multiprocessing data transfer issues

import os
import re
import csv
import zipfile
import io
from datetime import datetime
import multiprocessing
import functools
import argparse
import glob

# --- Constants ---
START_DATE = "20130101"
LOCAL_BASE_DIR = '../ORATS'
ZIP_FILES_DIR = os.path.join(LOCAL_BASE_DIR, 'zip')
TICKERS_FILE = os.path.join(LOCAL_BASE_DIR, 'tickers_of_interest.csv')
TRADE_DATE_COLUMN_NAME = "trade_date"
ZIP_DATE_FORMAT = "%Y%m%d"
INPUT_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y"]

def load_bad_dates(ticker, base_dir, start_date_filter=None, end_date_filter=None):
    """Loads a set of bad dates for a given ticker, optionally filtered by a date range."""
    bad_dates = set()
    bad_dates_file = os.path.join(base_dir, f"{ticker}_BADDATES.csv")
    print(f"    - Checking for bad dates file: {os.path.basename(bad_dates_file)}")
    if os.path.exists(bad_dates_file):
        try:
            with open(bad_dates_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        date_str = row[0].strip()
                        if start_date_filter and end_date_filter:
                            if start_date_filter <= date_str <= end_date_filter:
                                bad_dates.add(date_str)
                        else:
                            bad_dates.add(date_str)
            print(f"      -> Found and loaded {len(bad_dates)} bad dates.")
        except Exception as e:
            print(f"Error reading bad dates file for {ticker}: {e}")
    else:
        print("      -> File not found.")
    return bad_dates

def read_tickers(filepath):
    """Reads tickers of interest from the specified file."""
    print(f"Reading tickers from {filepath}...")
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            tickers = {row[0].strip() for row in csv.reader(file) if row and row[0].strip()}
            print(f" -> Found {len(tickers)} tickers.")
            return tickers
    except Exception as e:
        print(f"Error reading tickers from {filepath}: {e}")
        return set()

def read_existing_trade_dates(ticker, base_dir, start_date_filter=None, end_date_filter=None):
    """Reads a ticker's CSV to find which trade dates have already been processed, optionally filtered by date range."""
    trade_dates = set()
    ticker_filepath = os.path.join(base_dir, f"{ticker}.csv")
    print(f"    - Checking for existing data file: {os.path.basename(ticker_filepath)}")
    if not os.path.exists(ticker_filepath):
        print("      -> File not found. No existing dates to load.")
        return trade_dates
    try:
        with open(ticker_filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if TRADE_DATE_COLUMN_NAME not in reader.fieldnames:
                print(f"      -> WARNING: '{TRADE_DATE_COLUMN_NAME}' not in header. Cannot read existing dates.")
                return trade_dates
            for row in reader:
                date_val = row.get(TRADE_DATE_COLUMN_NAME, "").strip()
                if date_val:
                    for fmt in INPUT_DATE_FORMATS:
                        try:
                            parsed_date_str = datetime.strptime(date_val, fmt).strftime(ZIP_DATE_FORMAT)
                            if start_date_filter and end_date_filter:
                                if start_date_filter <= parsed_date_str <= end_date_filter:
                                    trade_dates.add(parsed_date_str)
                            else:
                                trade_dates.add(parsed_date_str)
                            break
                        except ValueError:
                            continue
        print(f"      -> Found {len(trade_dates)} previously processed trade dates in range.")
    except Exception as e:
        print(f"Error reading trade dates from {ticker_filepath}: {e}")
    return trade_dates

def extract_trade_date_from_filename(zip_filename):
    """Extracts YYYYMMDD date from a ZIP filename."""
    match = re.search(r'(\d{8})\.zip$', zip_filename, re.IGNORECASE)
    return match.group(1) if match else None

def worker_process_zip(zip_info, tickers_of_interest, existing_trade_dates, bad_dates_by_ticker):
    """
    WORKER FUNCTION: Processes a single ZIP, writing any found data to temporary partial files.
    This avoids returning large data objects to the main process.
    """
    zip_filepath, trade_date = zip_info
    basename = os.path.basename(zip_filepath)

    needed_tickers = {
        ticker for ticker in tickers_of_interest
        if trade_date not in existing_trade_dates.get(ticker, set())
        and trade_date not in bad_dates_by_ticker.get(ticker, set())
    }

    if not needed_tickers:
        return f"Skipped {basename}: All tickers already processed or have bad dates."

    collected_data = {}
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for csv_name in [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]:
                with zip_ref.open(csv_name) as csv_file:
                    text_file = io.TextIOWrapper(csv_file, 'utf-8')
                    reader = csv.DictReader(text_file)
                    fieldnames = reader.fieldnames
                    ticker_col = next((col for col in fieldnames if col.strip().lower() == 'ticker'), None)
                    if not ticker_col: continue

                    for row in reader:
                        ticker_val = row.get(ticker_col, "").strip()
                        if ticker_val in needed_tickers:
                            if ticker_val not in collected_data:
                                collected_data[ticker_val] = {'fieldnames': fieldnames, 'rows': []}
                            collected_data[ticker_val]['rows'].append(row)

        # Write collected data to temporary partial files
        for ticker, data in collected_data.items():
            if not data['rows']: continue
            # Create a unique partial file for this ticker and date
            part_filepath = os.path.join(LOCAL_BASE_DIR, f"{ticker}_{trade_date}.csv.part")
            with open(part_filepath, 'w', newline='', encoding='utf-8') as part_file:
                writer = csv.DictWriter(part_file, fieldnames=data['fieldnames'])
                writer.writeheader()
                writer.writerows(data['rows'])
        
        row_counts = {t: len(d['rows']) for t, d in collected_data.items() if d['rows']}
        if not row_counts:
            return f"Processed {basename}: No new data found for tickers of interest."
        return f"Processed {basename}: Found {row_counts}"

    except Exception as e:
        return f"ERROR processing {basename}: {e}"

def consolidate_partial_files(tickers):
    """Finds all .part files, consolidates them into the main CSVs, and cleans them up."""
    print("\n--- Phase: Consolidating Temporary Files ---")
    for ticker in sorted(list(tickers)):
        print(f"\nConsolidating for ticker: {ticker}")
        partial_files = sorted(glob.glob(os.path.join(LOCAL_BASE_DIR, f"{ticker}_*.csv.part")))
        if not partial_files:
            print("  -> No new partial files found to consolidate.")
            continue

        print(f"  -> Found {len(partial_files)} partial file(s) to process.")
        master_filepath = os.path.join(LOCAL_BASE_DIR, f"{ticker}.csv")
        master_file_exists = os.path.exists(master_filepath)
        
        total_rows_written = 0
        try:
            with open(master_filepath, 'a', newline='', encoding='utf-8') as master_outfile:
                # Get the header from the first partial file.
                with open(partial_files[0], 'r', newline='', encoding='utf-8') as p_file:
                    reader = csv.reader(p_file)
                    header = next(reader)
                
                writer = csv.writer(master_outfile)
                if not master_file_exists or os.path.getsize(master_filepath) == 0:
                    writer.writerow(header)

                # Append data from all partial files
                for part_file in partial_files:
                    with open(part_file, 'r', newline='', encoding='utf-8') as p_file:
                        reader = csv.reader(p_file)
                        next(reader) # Skip header of partial file
                        rows_to_write = list(reader)
                        writer.writerows(rows_to_write)
                        total_rows_written += len(rows_to_write)
                    os.remove(part_file) # Clean up the partial file
        except Exception as e:
            print(f"  -> ERROR consolidating files for {ticker}: {e}")
            continue

        if total_rows_written > 0:
            print(f"  -> Wrote {total_rows_written} new rows to {os.path.basename(master_filepath)} from {len(partial_files)} partial file(s).")

def main():
    """Main function to coordinate the parallel ORATS data processing workflow."""
    parser = argparse.ArgumentParser(description="Process ORATS data ZIP files in parallel.")
    parser.add_argument('--d', '--dates', nargs=2, metavar=('YYYYMMDD_START', 'YYYYMMDD_END'),
                        help="An optional start and end date (inclusive) to filter ZIP files.")
    args = parser.parse_args()

    print("--- Phase: Initialization ---")
    tickers = read_tickers(TICKERS_FILE)
    if not tickers:
        print("No tickers of interest found. Exiting.")
        return
    print(f"Processing will be done for tickers: {', '.join(sorted(tickers))}")

    print("\n--- Phase: Pre-computation of Processed Dates ---")
    start_range, end_range = (args.d[0], args.d[1]) if args.d else (None, None)
    
    ticker_trade_dates = {}
    bad_dates_by_ticker = {}
    for t in sorted(list(tickers)):
        print(f"\nLoading metadata for ticker: {t}")
        ticker_trade_dates[t] = read_existing_trade_dates(t, LOCAL_BASE_DIR, start_range, end_range)
        bad_dates_by_ticker[t] = load_bad_dates(t, LOCAL_BASE_DIR, start_range, end_range)
    
    if not os.path.isdir(ZIP_FILES_DIR):
        print(f"\nError: Input directory '{ZIP_FILES_DIR}' not found.")
        return

    print("\n--- Phase: Scanning and Filtering Input ZIP files ---")
    all_zip_filenames = [f for f in os.listdir(ZIP_FILES_DIR) if f.lower().endswith('.zip')]
    
    zip_files_to_process = []
    date_filter = START_DATE if not args.d else start_range
    filter_mode = f"on or after the hardcoded START_DATE: {START_DATE}" if not args.d else f"between {start_range} and {end_range}"

    print(f"Filtering {len(all_zip_filenames)} total ZIP files with dates {filter_mode}.")
    for zip_file in all_zip_filenames:
        trade_date = extract_trade_date_from_filename(zip_file)
        if trade_date and trade_date >= date_filter:
            if args.d and trade_date > end_range:
                continue
            zip_files_to_process.append((os.path.join(ZIP_FILES_DIR, zip_file), trade_date))

    zip_files_to_process.sort(key=lambda x: x[1])
    
    if not zip_files_to_process:
        print("\nNo new ZIP files found to process in the specified date range. Exiting.")
        return
        
    print(f"\n--- Phase: Parallel Processing ---")
    print(f"Found {len(zip_files_to_process)} ZIP files to analyze in '{os.path.basename(ZIP_FILES_DIR)}'.")

    num_workers = multiprocessing.cpu_count()
    print(f"Initializing a pool with {num_workers} worker processes...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_func = functools.partial(
            worker_process_zip,
            tickers_of_interest=tickers,
            existing_trade_dates=ticker_trade_dates,
            bad_dates_by_ticker=bad_dates_by_ticker
        )
        print("Distributing tasks to worker processes...")
        results = pool.map(worker_func, zip_files_to_process)

    print("\nParallel processing phase complete. Worker results:")
    # Print status messages from workers
    for status in results:
        if status:
            print(f"  - {status}")
    
    consolidate_partial_files(tickers)
    
    print("\n\n--- All Phases Complete ---")

if __name__ == '__main__':
    main()
