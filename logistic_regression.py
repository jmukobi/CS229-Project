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
    data = r.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds='24_7')
    if not data:
        print(f"No data found for {symbol}")
        return None
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['begins_at'])
    df.set_index('timestamp', inplace=True)
    df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
    df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce')
    df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    return df.dropna()

def create_profit_labels(df, profit_margin=PROFIT_MARGIN, future_window=FUTURE_WINDOW):
    """Creates labels for predicting profitable trades."""
    df['future_price'] = df['close_price'].shift(-future_window)
    df['profit_threshold'] = df['close_price'] * (1 + profit_margin)
    df['label'] = (df['future_price'] > df['profit_threshold']).astype(int)
    return df.dropna()

def add_technical_indicators(df):
    """Calculates a comprehensive set of technical indicators using pandas_ta."""
    try:
        # Calculate Simple Moving Averages (SMA)
        df['SMA_10'] = ta.sma(df['close_price'], length=10)
        df['SMA_50'] = ta.sma(df['close_price'], length=50)

        # Exponential Moving Averages (EMA)
        df['EMA_10'] = ta.ema(df['close_price'], length=10)
        df['EMA_50'] = ta.ema(df['close_price'], length=50)

        # Relative Strength Index (RSI)
        df['RSI_14'] = ta.rsi(df['close_price'], length=14)
        df['RSI_28'] = ta.rsi(df['close_price'], length=28)

        # MACD and MACD Signal
        macd = ta.macd(df['close_price'])
        if macd is not None and not macd.empty:
            df['MACD'] = macd.get('MACD_12_26_9', np.nan)
            df['MACD_Signal'] = macd.get('MACDs_12_26_9', np.nan)
        else:
            print("MACD calculation failed; filling with NaN.")
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan

        # Bollinger Bands
        bbands = ta.bbands(df['close_price'], length=20)
        if bbands is not None and not bbands.empty:
            df['Bollinger_Upper'] = bbands.get('BBU_20_2.0', np.nan)
            df['Bollinger_Lower'] = bbands.get('BBL_20_2.0', np.nan)
        else:
            print("Bollinger Bands calculation failed; filling with NaN.")
            df['Bollinger_Upper'] = np.nan
            df['Bollinger_Lower'] = np.nan

        # Stochastic Oscillator
        stoch = ta.stoch(df['high_price'], df['low_price'], df['close_price'])
        if stoch is not None and not stoch.empty:
            df['Stoch'] = stoch.get('STOCHk_14_3_3', np.nan)
        else:
            print("Stochastic Oscillator calculation failed; filling with NaN.")
            df['Stoch'] = np.nan

        # Average True Range (ATR)
        df['ATR_14'] = ta.atr(df['high_price'], df['low_price'], df['close_price'], length=14)

        # Commodity Channel Index (CCI)
        df['CCI_20'] = ta.cci(df['high_price'], df['low_price'], df['close_price'], length=20)

        # Williams %R
        df['Williams_%R'] = ta.willr(df['high_price'], df['low_price'], df['close_price'], length=14)

        # On-Balance Volume (OBV)
        df['OBV'] = ta.obv(df['close_price'], df['volume'])

        # Momentum (MOM)
        df['Momentum_10'] = ta.mom(df['close_price'], length=10)

        # Drop rows with NaN values after adding indicators
        df.dropna(inplace=True)

    except Exception as e:
        print(f"Error calculating technical indicators: {e}")

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
    """Makes predictions using the trained model."""
    df = add_technical_indicators(df)
    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)
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
