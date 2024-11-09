import robin_stocks.robinhood as r
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import traceback
from configuration import username, password


# Constants
SYMBOL = "ETH"
DATA_INTERVAL = "hour"
DATA_SPAN = "month"
PROFIT_MARGIN = 0.005
FUTURE_WINDOW = 12

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
    """Fetches historical data for a given cryptocurrency."""
    try:
        data = r.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds='24_7')
        if not data:
            print(f"No data found for {symbol}")
            return None
        
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

def create_profit_labels(df, profit_margin=PROFIT_MARGIN, future_window=FUTURE_WINDOW):
    """Creates labels for predicting profitable trades."""
    df['future_price'] = df['close_price'].shift(-future_window)
    df['profit_threshold'] = df['close_price'] * (1 + profit_margin)
    df['label'] = (df['future_price'] > df['profit_threshold']).astype(int)
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    """Calculates technical indicators using pandas_ta."""
    # Calculate Simple Moving Average (SMA)
    df['SMA_10'] = ta.sma(df['close_price'], length=10)
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA_10'] = ta.ema(df['close_price'], length=10)
    
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.rsi(df['close_price'], length=14)
    
    # Calculate MACD
    macd = ta.macd(df['close_price'])
    df['MACD'] = macd['MACD_12_26_9']
    
    # Calculate Bollinger Bands
    bbands = ta.bbands(df['close_price'], length=20)
    
    # Dynamically extract the columns for Bollinger Bands
    if bbands is not None:
        upper_band = [col for col in bbands.columns if 'BBU' in col][0]
        lower_band = [col for col in bbands.columns if 'BBL' in col][0]
        df['Bollinger_Upper'] = bbands[upper_band]
        df['Bollinger_Lower'] = bbands[lower_band]
    else:
        print("Bollinger Bands columns not found. Using default values.")
        df['Bollinger_Upper'] = pd.NA
        df['Bollinger_Lower'] = pd.NA
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def normalize_features(df, feature_cols):
    """Normalizes the features using StandardScaler."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def train_logistic_regression(df):
    """Trains a logistic regression model with class weight adjustment."""
    feature_cols = ['SMA_10', 'EMA_10', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    df = df[feature_cols + ['label']].dropna()

    # Separate features (X) and labels (y)
    X = df[feature_cols]
    y = df['label']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the logistic regression model with class weight adjustment
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    
    return model

def check_label_distribution(df):
    """Prints the distribution of labels in the dataset."""
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)
        print(f"\nPercentage of Profitable Trades (label=1): {label_counts.get(1, 0) / len(df) * 100:.2f}%")
    else:
        print("Labels not found in the dataset.")

def main():
    """Main function to run the pipeline."""
    try:
        login()
        df = get_crypto_historical_data()
        if df is not None:
            df = create_profit_labels(df)
            # Check the label distribution
            check_label_distribution(df)
            df = add_technical_indicators(df)
            
            if not df.empty:
                train_logistic_regression(df)
        
        logout()
    except Exception as e:
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        print(f"An error occurred:\n{tb_str}")

if __name__ == "__main__":
    main()
