import robin_stocks.robinhood as r
import pandas as pd
import time
import traceback
from configuration import username, password

# Constants
SYMBOL = "ETH"
DATA_INTERVAL = "10minute"
DATA_SPAN = "week"

def login():
    """
    Logs into Robinhood using credentials from configuration file.
    """
    try:
        r.login(username, password, expiresIn=86400000, by_sms=True)
        print("Logged in to Robinhood")
    except Exception as e:
        print("Failed to log in:", e)
        exit()

def logout():
    """
    Logs out of Robinhood session.
    """
    r.logout()
    print("Logged out of Robinhood")

def get_crypto_historical_data(symbol=SYMBOL, interval=DATA_INTERVAL, span=DATA_SPAN):
    """
    Fetches historical data for a given cryptocurrency.
    :param symbol: Symbol of the cryptocurrency (e.g., 'ETH', 'BTC')
    :param interval: Data interval ('5minute', '10minute', 'hour', etc.)
    :param span: Time span ('day', 'week', 'month', etc.)
    :return: DataFrame with historical price data
    """
    try:
        data = r.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds='24_7')
        if not data:
            print(f"No data found for {symbol}")
            return None
        
        # Convert to DataFrame and process
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['begins_at'])
        df.set_index('timestamp', inplace=True)
        df['close_price'] = df['close_price'].astype(float)
        print(f"Fetched {len(df)} data points for {symbol}")
        return df
    except Exception as e:
        print("Error fetching data:", e)
        return None

def save_data_to_csv(df, filename='crypto_data.csv'):
    """
    Saves the DataFrame to a CSV file.
    :param df: DataFrame containing historical data
    :param filename: Name of the CSV file to save to
    """
    if df is not None:
        df.to_csv(filename)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

def main():
    """
    Main function to log in, fetch data, and save it to a CSV file.
    """
    try:
        login()
        data = get_crypto_historical_data()
        if data is not None:
            save_data_to_csv(data)
        logout()
    except Exception as e:
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        print(f"An error occurred:\n{tb_str}")

if __name__ == "__main__":
    main()
