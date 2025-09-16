# ORATS Data Downloader
# Downloads historical options data from ORATS FTP server
# Only grabs files we don't already have locally
#
#
# Setup:
# Set these environment variables before running:
#   export ORATS_FTP_USER="your_username"
#   export ORATS_FTP_PASSWORD="your_password"
#
# Usage:
#   python 0020_fetch_ORATS_option_data.py                      # get recent missing files
#   python 0020_fetch_ORATS_option_data.py --d 20250101 20250630  # specific date range
#
# Features:
# - Skips files already downloaded (incremental updates)
# - Filters out bad dates from {TICKER}_BADDATES.csv files
# - Handles FTP errors gracefully
# - Can limit to specific date ranges
import ftplib
import os
import datetime
import re
import csv
import argparse



# FTP server details
FTP_HOST = 'orats.hostedftp.com'
FTP_BASE_DIR = '/BEc4UffhfcWHc9l9WBUaqXByk'
LOCAL_BASE_DIR = '../ORATS/zip'

# Configuration for date lookback
LOOKBACK_DAYS = 10000  # Default lookback if no date range is specified
YEAR_THRESHOLD = 2006

# Regex patterns for file parsing
TRADE_DATE_REGEX = re.compile(r'(\d{8})')
TICKER_REGEX = re.compile(r'^([A-Za-z0-9]+)_\d{8}\.zip$')

# Cache for bad dates by ticker
BAD_DATES_CACHE = {}


def load_bad_dates(ticker):
    """
    Load bad dates for a specific ticker from its BADDATES.csv file.
    Returns a set of datetime.date objects representing dates to avoid.
    """
    if ticker in BAD_DATES_CACHE:
        return BAD_DATES_CACHE[ticker]

    bad_dates = set()
    bad_dates_file = os.path.join(LOCAL_BASE_DIR, f"{ticker}_BADDATES.csv")

    if os.path.exists(bad_dates_file):
        try:
            with open(bad_dates_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        date_str = row[0].strip()
                        try:
                            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                            bad_dates.add(date_obj)
                        except ValueError:
                            print(f"Warning: Invalid date format in {bad_dates_file}: {date_str}")
            print(f"Loaded {len(bad_dates)} bad dates for {ticker}")
        except Exception as e:
            print(f"Error reading bad dates file for {ticker}: {e}")

    BAD_DATES_CACHE[ticker] = bad_dates
    return bad_dates


def extract_ticker(filename):
    """
    Extracts the ticker symbol from the given filename.
    Returns the ticker or None if no valid ticker is found.
    """
    match = TICKER_REGEX.match(filename)
    if match:
        return match.group(1).upper()
    return None


def extract_trade_date(filename):
    """
    Extracts the trade date (YYYYMMDD) from the given filename using regex.
    Returns a datetime object or None if no valid date is found.
    """
    match = TRADE_DATE_REGEX.search(filename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    return None


def should_download_file(filename, start_date=None, end_date=None):
    """
    Determines if a file should be downloaded based on several criteria.
    """
    if not filename.lower().endswith('.zip'):
        return False

    trade_date = extract_trade_date(filename)
    if not trade_date:
        return False

    # Date range filtering
    if start_date and trade_date < start_date:
        return False
    if end_date and trade_date > end_date:
        return False

    # Default lookback if no date range
    if not start_date and not end_date:
        cutoff_date = datetime.date.today() - datetime.timedelta(days=LOOKBACK_DAYS)
        if trade_date < cutoff_date:
            return False

    # Check against bad dates
    ticker = extract_ticker(filename)
    if ticker:
        bad_dates = load_bad_dates(ticker)
        if trade_date in bad_dates:
            print(f"Skipping bad date {trade_date} for {ticker}")
            return False

    return True


def traverse_ftp(ftp, local_dir, start_date=None, end_date=None):
    """
    Recursively traverse the FTP server and download missing files.
    """
    current_dir = ftp.pwd()
    try:
        items = ftp.nlst()
    except ftplib.error_perm as e:
        print(f"Could not list directory {current_dir}: {e}")
        return

    for item in items:
        if item in ['.', '..']:
            continue

        try:
            ftp.cwd(item)
            new_dir = ftp.pwd()

            if item.isdigit():
                year = int(item)
                if year <= YEAR_THRESHOLD:
                    print(f"Skipping directory {item} (year <= {YEAR_THRESHOLD})")
                    ftp.cwd(current_dir)
                    continue

            print(f"Entering directory: {new_dir}")
            traverse_ftp(ftp, local_dir, start_date, end_date)
            ftp.cwd(current_dir)

        except ftplib.error_perm:
            if should_download_file(item, start_date, end_date):
                local_file_path = os.path.join(local_dir, item)
                if os.path.exists(local_file_path):
                    print(f"Skipping existing file: {local_file_path}")
                else:
                    ticker = extract_ticker(item) or "UNKNOWN"
                    date = extract_trade_date(item) or "UNKNOWN"
                    print(f"Downloading file for {ticker} on {date}: {item} from {current_dir}")
                    try:
                        with open(local_file_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {item}', f.write)
                    except Exception as e:
                        print(f"Error downloading {item}: {e}")
            continue


def download_missing_zip_files(start_date=None, end_date=None):
    """
    Main function to orchestrate the FTP connection and download process.
    """
    import os

    # Get credentials from environment variables
    ftp_user = os.environ.get('ORATS_FTP_USER')
    ftp_password = os.environ.get('ORATS_FTP_PASSWORD')

    if not ftp_user or not ftp_password:
        raise ValueError("Missing FTP credentials. Please set ORATS_FTP_USER and ORATS_FTP_PASSWORD environment variables.")

    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(user=ftp_user, passwd=ftp_password)
    print(f"Connected to {FTP_HOST}")

    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

    traverse_ftp(ftp, LOCAL_BASE_DIR, start_date, end_date)

    ftp.quit()
    print("\nAll downloads complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download missing ORATS data files from an FTP server.")
    parser.add_argument('--d', nargs=2, metavar=('YYYYMMDD', 'YYYYMMDD'),
                        help='Specify a start and end date for data download (inclusive).')
    args = parser.parse_args()

    start_date, end_date = None, None
    if args.d:
        try:
            start_date = datetime.datetime.strptime(args.d[0], "%Y%m%d").date()
            end_date = datetime.datetime.strptime(args.d[1], "%Y%m%d").date()
            print(f"Date range specified: {start_date} to {end_date}")
        except ValueError:
            print("Error: Invalid date format. Please use YYYYMMDD.")
            parser.print_help()
            exit(1)

    download_missing_zip_files(start_date, end_date)
