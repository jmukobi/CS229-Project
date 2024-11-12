import numpy as np
import robin_stocks.robinhood as r
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import traceback
from configuration import username, password

# Constants
SYMBOL = "ETH"
DATA_INTERVAL = "hour"
DATA_SPAN = "month"
PROFIT_MARGIN = 0.01
FUTURE_WINDOW = 24

FEATURE_COLS = [
    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
    'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
    'Bollinger_Upper', 'Bollinger_Lower',
    'Stoch', 'ATR_14', 'CCI_20', 'Williams_%R',
    'OBV', 'Momentum_10'
]

def login():
    """Logs into Robinhood."""
    try:
        r.login(username, password, expiresIn=86400000, by_sms=True)
        print("Logged in to Robinhood")
    except Exception as e:
        print("Failed to log in:", e)
        exit()

def logout():
    """Logs out of Robinhood."""
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
        df.set_index('timestamp', inplace=False)  # Ensure 'timestamp' column is not dropped
        
        # Convert price columns to numeric
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce')
        df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce')
        df['open_price'] = pd.to_numeric(df['open_price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        print(f"Fetched {len(df)} data points for {symbol}")
        return df.dropna().reset_index()  # Preserve the 'timestamp' column
    except Exception as e:
        print("Error fetching data:", e)
        return None

def create_profit_labels(df, profit_margin=PROFIT_MARGIN, future_window=FUTURE_WINDOW):
    """Creates labels for predicting profitable trades."""
    df['future_price'] = df['close_price'].shift(-future_window)
    df['profit_threshold'] = df['close_price'] * (1 + profit_margin)
    df['label'] = (df['future_price'] > df['profit_threshold']).astype(int)
    return df.dropna()

def add_technical_indicators(df):
    """Calculates a comprehensive set of technical indicators using pandas_ta."""
    df.loc[:, 'SMA_10'] = ta.sma(df['close_price'], length=10)
    df.loc[:, 'SMA_50'] = ta.sma(df['close_price'], length=50)
    df.loc[:, 'EMA_10'] = ta.ema(df['close_price'], length=10)
    df.loc[:, 'EMA_50'] = ta.ema(df['close_price'], length=50)
    df.loc[:, 'RSI_14'] = ta.rsi(df['close_price'], length=14)
    df.loc[:, 'RSI_28'] = ta.rsi(df['close_price'], length=28)

    # Handle potential None values for MACD and others
    macd = ta.macd(df['close_price'])
    if macd is not None:
        df.loc[:, 'MACD'] = macd['MACD_12_26_9']
        df.loc[:, 'MACD_Signal'] = macd['MACDs_12_26_9']
    else:
        df.loc[:, 'MACD'] = np.nan
        df.loc[:, 'MACD_Signal'] = np.nan
    
    bbands = ta.bbands(df['close_price'], length=20)
    if bbands is not None:
        df.loc[:, 'Bollinger_Upper'] = bbands.get('BBU_20_2.0', np.nan)
        df.loc[:, 'Bollinger_Lower'] = bbands.get('BBL_20_2.0', np.nan)
    
    df.loc[:, 'Stoch'] = ta.stoch(df['high_price'], df['low_price'], df['close_price']).get('STOCHk_14_3_3', np.nan)
    df.loc[:, 'ATR_14'] = ta.atr(df['high_price'], df['low_price'], df['close_price'], length=14)
    df.loc[:, 'CCI_20'] = ta.cci(df['high_price'], df['low_price'], df['close_price'], length=20)
    df.loc[:, 'Williams_%R'] = ta.willr(df['high_price'], df['low_price'], df['close_price'], length=14)
    df.loc[:, 'OBV'] = ta.obv(df['close_price'], df['volume'])
    df.loc[:, 'Momentum_10'] = ta.mom(df['close_price'], length=10)
    
    # Drop rows with NaN values after adding indicators
    df.dropna(inplace=True)
    
    return df


def train_model(df):
    """
    Trains a logistic regression model on the given DataFrame.
    """
    feature_cols = [
        'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
        'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower', 'Stoch', 'ATR_14',
        'CCI_20', 'Williams_%R', 'OBV', 'Momentum_10'
    ]
    
    df = df[feature_cols + ['label']].dropna()

    # Separate features (X) and labels (y)
    X = df[feature_cols]
    y = df['label']

    # Check if we have at least two classes
    if len(y.unique()) < 2:
        print("Skipping training: only one class present in the data.")
        return None, None

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the logistic regression model with class weight adjustment
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler


def make_prediction(df, model, scaler):
    """
    Makes predictions on the given DataFrame using the trained model and scaler.
    """
    # Define the feature columns
    feature_cols = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
                    'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
                    'Bollinger_Upper', 'Bollinger_Lower', 'Stoch', 'ATR_14',
                    'CCI_20', 'Williams_%R', 'OBV', 'Momentum_10']
    
    # Add technical indicators to the DataFrame
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Check if there's enough data left
    if df.empty:
        print("No data left after adding indicators and dropping NaNs. Skipping prediction.")
        return None  # Return None if no data is left

    # Prepare features for prediction
    X = df[feature_cols]
    
    # Check if X is empty before scaling
    if X.empty:
        print("No valid data points to scale. Skipping prediction.")
        return None

    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions using the model
    predictions = model.predict(X_scaled)
    return predictions


def main():
    login()
    df = get_crypto_historical_data()
    if df is not None:
        df = create_profit_labels(df)
        df = add_technical_indicators(df)
        model, scaler = train_model(df)
        predictions = make_prediction(df, model, scaler)
        print("Predictions:", predictions)
    logout()

if __name__ == "__main__":
    main()
