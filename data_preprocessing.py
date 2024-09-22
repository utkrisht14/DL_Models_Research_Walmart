import numpy as np
import pandas as pd
import yfinance as yf
import fredapi
import holidays

fed = fredapi.Fred(api_key="API_KEY")

# Download historical data for Walmart Inc.
ticker = 'WMT'
walmart_data = yf.download(ticker, start="1973-01-01", end="2024-09-05")

# Save the data to a CSV file
walmart_data.to_csv("walmart_share_price.csv")

# Display the first few rows
walmart_data.head()

# Specify the date till the data is required
start_date = "1973-01-01"
end_date = "2024-09-05"

# Interest Rate Data
interest_rate_data = fed.get_series("FEDFUNDS", start_date="1973-01-01", end_date="2024-09-05")

# CPI Data
cpi_data = fed.get_series("CPIAUCSL", start_date=start_date, end_date=end_date)

# Unemployment Data
unemployment_rate_data = fed.get_series("UNRATE", start_date=start_date, end_date=end_date)

# GDP Growth Rate Data
gdp_growth_data = fed.get_series('A191RL1Q225SBEA', start_date=start_date, end_date=end_date)

# Consumer Confidence Index Data
cci_data = fed.get_series('UMCSENT', start_date=start_date, end_date=end_date)

# Retrieve gold prices 
gold_price_data = yf.download('GC=F', start=start_date, end=end_date)

# Retrieve crude oil prices 
oil_price_data = yf.download('CL=F', start=start_date, end=end_date)

# Exchange Rate Data
exchange_rate_data = yf.download('USDCHF=X', start=start_date, end=end_date)

# VIX data
vix_data = yf.download('^VIX', start=start_date, end=end_date)

# Define holidays (Black Friday, Christmas etc.)
black_friday_dates = pd.to_datetime([
    '1973-11-23', '1974-11-29', '1975-11-28', '1976-11-26',
    '1977-11-25', '1978-11-24', '1979-11-23', '1980-11-28',
    '1981-11-27', '1982-11-26', '1983-11-25', '1984-11-23',
    '1985-11-29', '1986-11-28', '1987-11-27', '1988-11-25',
    '1989-11-24', '1990-11-23', '1991-11-29', '1992-11-27',
    '1993-11-26', '1994-11-25', '1995-11-24', '1996-11-29',
    '1997-11-28', '1998-11-27', '1999-11-26', '2000-11-24',
    '2001-11-23', '2002-11-29', '2003-11-28', '2004-11-26',
    '2005-11-25', '2006-11-24', '2007-11-23', '2008-11-28',
    '2009-11-27', '2010-11-26', '2011-11-25', '2012-11-23',
    '2013-11-29', '2014-11-28', '2015-11-27', '2016-11-25',
    '2017-11-24', '2018-11-23', '2019-11-29', '2020-11-27',
    '2021-11-26', '2022-11-25', '2023-11-24', '2024-11-29'
])

christmas_dates = pd.to_datetime([
    '1973-12-25', '1974-12-25', '1975-12-25', '1976-12-25',
    '1977-12-25', '1978-12-25', '1979-12-25', '1980-12-25',
    '1981-12-25', '1982-12-25', '1983-12-25', '1984-12-25',
    '1985-12-25', '1986-12-25', '1987-12-25', '1988-12-25',
    '1989-12-25', '1990-12-25', '1991-12-25', '1992-12-25',
    '1993-12-25', '1994-12-25', '1995-12-25', '1996-12-25',
    '1997-12-25', '1998-12-25', '1999-12-25', '2000-12-25',
    '2001-12-25', '2002-12-25', '2003-12-25', '2004-12-25',
    '2005-12-25', '2006-12-25', '2007-12-25', '2008-12-25',
    '2009-12-25', '2010-12-25', '2011-12-25', '2012-12-25',
    '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25',
    '2017-12-25', '2018-12-25', '2019-12-25', '2020-12-25',
    '2021-12-25', '2022-12-25', '2023-12-25', '2024-12-25'
])

# List of historical U.S. presidential election dates
election_dates = pd.to_datetime([
    '1972-11-07', '1976-11-02', '1980-11-04', '1984-11-06',
    '1988-11-08', '1992-11-03', '1996-11-05', '2000-11-07',
    '2004-11-02', '2008-11-04', '2012-11-06', '2016-11-08',
    '2020-11-03'
])

# Calculate the day after each election result
day_after_election_dates = election_dates + pd.Timedelta(days=1)

# Load stock market data
df = pd.read_csv('walmart_share_price.csv', index_col='Date', parse_dates=True)

# Add day_before_weekend feature (Fridays)
df["day_before_weekend"] = (df.index.weekday == 4).astype(int)  # 4 represents Friday

# Combine all holidays into one list using numpy.concatenate
all_holidays = np.concatenate([black_friday_dates, christmas_dates])

# Create a day_before_holiday and day_after_holiday feature by shifting the holiday dates
day_before_holidays = all_holidays - pd.Timedelta(days=1)
day_after_holidays = all_holidays + pd.Timedelta(days=1)

# Mark if a day is the day before a holiday
df["day_before_holiday"] = df.index.isin(day_before_holidays).astype(int)

# Mark if a day is the day after a holiday
df["day_after_holiday"] = df.index.isin(day_after_holidays).astype(int)

# Mark if a day is the day after a weekend (Monday)
df["day_after_weekend"] = (df.index.weekday == 0).astype(int)  # 0 represents Monday

# Make some data into dataframe for easy merge.

unemployment_df = pd.DataFrame(unemployment_rate_data, columns=["Unemployment_Rate"])
cci_df = pd.DataFrame(cci_data, columns=["CCI"])
cpi_df = pd.DataFrame(cpi_data, columns=["CPI"])
gdp_df = pd.DataFrame(gdp_growth_data, columns=["GPD_Growth_Rate"])

# Now downloaded data from the yahoo are already in dataframe format. We need only `Adj Close` from it.

# Gold Rate
gold_df = gold_price_data[["Adj Close"]]
gold_df.columns = ["Gold_Price"]

# Oil Price
oil_df = oil_price_data[["Adj Close"]]
oil_df.columns = ["Oil_Price"]

# Exchange Rate
exchange_df = pd.DataFrame(unemployment_rate_data, columns=["Unemployment_Rate"])
exchange_df.columns = ["Exchange_Rate_Price"]

# VIX Data
vix_df = vix_data[["Adj Close"]]
vix_df.columns = ["VIX"]


# Make a function to merge all the data into a dataframe.
def custom_fill_all(df, cpi_df, unemployment_df, cci_df):
    # Merging CPI, unemployment, and CCI data with the main dataframe
    merged_df = df.join([cpi_df, unemployment_df, cci_df, gdp_df, oil_df, gold_df, exchange_df, vix_df], how='left')

    # Backward-fill for initial missing values before the first available data point
    merged_df.bfill(inplace=True)

    # Forward-fill for all remaining missing values after encountering the first available data point
    merged_df.ffill(inplace=True)

    return merged_df


merged_df = custom_fill_all(df, cpi_df, unemployment_df, cci_df)
print(merged_df.head())

print(merged_df.shape)

# Shift the target column to predict the next day's price
df["Adj Close Target"] = df["Adj Close"].shift(-1)


# Function to calculate RSI
def compute_rsi(series, window=14):
    # Calculate the price differences
    delta = series.diff()

    # Make two series: one for gains and one for losses 
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the rolling averages of the gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI based on RS
    rsi = 100 - (100 / (1 + rs))

    return rsi



merged_df["RSI"] = compute_rsi(merged_df["Adj Close"], window=14)

# Create the target column for the next day's Adjusted Close price
merged_df["Adj Close Target"] = merged_df['Adj Close'].shift(-1)

df_new = merged_df.copy()

df_new.drop(columns=["Adj Close", "Close"], inplace=True)

df_new.dropna(inplace=True)

print(df_new.shape)

# Set the index
df_new.reset_index(inplace=True)

# Save the data as new dataframe
df_new.to_csv("walmart_dataset.csv", index=False)
