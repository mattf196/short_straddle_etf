# Short Straddle ETF Analysis

## Usage

To generate the exhibits used in the paper (in /doc), run the programs in /src in numerical sequence:

```bash
python src/0010_fetch_cmt_yield_data.py
python src/0020_fetch_ORATS_option_data.py
python src/0030_process_ORATS_option_data.py
python src/0040_infer_dividend_yields.py
python src/0050_calibrate_SVI_models.py
python src/0051_visualize_SVI_calibrations.py
python src/0060_interpolate_fixed_option_tenors.py
python src/0070_track_option_performance.py
python src/0080_combine_option_performance.py
```


## Data Requirements

Input data from ORATS cannot be provided due to copyright restrictions. With an ORATS subscription, the programs (in particular 0020 and beyond) should run seamlessly.

export ORATS_FTP_USER="myusername@site.com"
export ORATS_FTP_PASSWORD="mypassword"


## Repository Structure

- `/src` - Analysis programs listed above (order matters; run in the ascending sequence shown above).
- `/doc` - Documentation and paper exhibits.   This contains a summary of our findings. 
- `/ORATS` - ORATS data directory.  This also contains the two necessary input files, trades_of_interest.csv and tickers_of_interest.csv, as well as intermediate and final output files.
