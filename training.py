import numpy as np
import robin_stocks.robinhood as r
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback
from configuration import username, password
from tqdm import tqdm

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
        
        # Convert price columns to numeric, forcing any errors to NaN
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce')
        df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce')
        df['open_price'] = pd.to_numeric(df['open_price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        print(f"Fetched {len(df)} data points for {symbol}")
        return df.dropna()  # Drop rows with NaN values after conversion
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
    """Calculates a comprehensive set of technical indicators using pandas_ta."""
    # Calculate Simple Moving Average (SMA) with multiple lengths
    df['SMA_10'] = ta.sma(df['close_price'], length=10)
    df['SMA_50'] = ta.sma(df['close_price'], length=50)
    
    # Exponential Moving Average (EMA) with multiple lengths
    df['EMA_10'] = ta.ema(df['close_price'], length=10)
    df['EMA_50'] = ta.ema(df['close_price'], length=50)
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = ta.rsi(df['close_price'], length=14)
    df['RSI_28'] = ta.rsi(df['close_price'], length=28)
    
    # MACD and MACD Signal
    macd = ta.macd(df['close_price'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close_price'], length=20)
    df['Bollinger_Upper'] = bbands['BBU_20_2.0']
    df['Bollinger_Lower'] = bbands['BBL_20_2.0']
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['high_price'], df['low_price'], df['close_price'])
    df['Stoch'] = stoch['STOCHk_14_3_3']
    
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
    
    return df

def normalize_features(df, feature_cols):
    """Normalizes the features using StandardScaler."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def train_logistic_regression(df):
    """Trains a logistic regression model with class weight adjustment."""
    feature_cols = [
        'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
        'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
        'Bollinger_Upper', 'Bollinger_Lower',
        'Stoch', 'ATR_14', 'CCI_20', 'Williams_%R',
        'OBV', 'Momentum_10'
    ]
    
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
    f1_score_value = f1_score(y_test, y_pred, zero_division=1)
    print(f"F1 Score: {f1_score_value:.2f}")
    
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


def grid_search(df, profit_margins, future_windows):
    """Perform grid search to find the best PROFIT_MARGIN and FUTURE_WINDOW."""
    best_score = 0
    best_params = None
    best_model = None
    best_scaler = None
    best_X_test = None
    best_y_test = None
    best_y_pred = None
    
    results = []

    # Updated feature columns list
    feature_cols = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
                    'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
                    'Bollinger_Upper', 'Bollinger_Lower', 'Stoch', 'ATR_14',
                    'CCI_20', 'Williams_%R', 'OBV', 'Momentum_10']

    # Calculate total number of iterations
    total_iterations = len(profit_margins) * len(future_windows)

    # Use tqdm for a progress bar
    with tqdm(total=total_iterations, desc="Grid Search Progress", unit="comb") as pbar:
        for margin in profit_margins:
            for window in future_windows:
                # Step 1: Create labels with the current parameters
                df_labeled = create_profit_labels(df.copy(), profit_margin=margin, future_window=window)
                df_labeled = add_technical_indicators(df_labeled)
                df_labeled.dropna(inplace=True)

                if len(df_labeled) < 10:
                    pbar.update(1)
                    continue

                X = df_labeled[feature_cols]
                y = df_labeled['label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression(class_weight='balanced', random_state=42)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                score = f1_score(y_test, y_pred, zero_division=1)
                
                results.append((margin, window, score))

                # Update best model if current score is better
                if score > best_score:
                    best_score = score
                    best_params = (margin, window)
                    best_model = model
                    best_scaler = scaler
                    best_X_test = X_test_scaled
                    best_y_test = y_test
                    best_y_pred = y_pred

                # Update progress bar
                pbar.update(1)

    # Plot the 3D surface with the best point
    if results:
        plot_3d_surface(results, best_params=best_params, best_score=best_score)
    
    if best_params:
        print(f"\nBest Parameters: PROFIT_MARGIN={best_params[0]}, FUTURE_WINDOW={best_params[1]}")
        print(f"Best F1 Score: {best_score:.2f}")

        accuracy = accuracy_score(best_y_test, best_y_pred)
        print(f"\nAccuracy: {accuracy:.2f}")
        print("Detailed Classification Report:")
        print(classification_report(best_y_test, best_y_pred, zero_division=1))

        # Visualize the buy signals on the historical price chart
        plot_buy_signals(df, best_model, best_scaler)

        # Extract and print feature weights for the best model
        weights = best_model.coef_[0]
        print("\nFeature Weights (Best Model):")
        for feature, weight in zip(feature_cols, weights):
            print(f"{feature}: {weight:.4f}")

    return best_params

def plot_3d_surface(results, best_params=None, best_score=None):
    """Plots a 3D surface of F1 scores based on profit margins and future windows, with the best point highlighted."""
    # Extract data
    margins, windows, scores = zip(*results)
    
    # Convert to numpy arrays for plotting
    margins = np.array(margins)
    windows = np.array(windows)
    scores = np.array(scores)

    # Create a grid for 3D plotting
    margins_grid, windows_grid = np.meshgrid(np.unique(margins), np.unique(windows))
    scores_grid = np.array([scores[i] for i in range(len(scores))]).reshape(margins_grid.shape)
    
    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(margins_grid, windows_grid, scores_grid, cmap='viridis', edgecolor='k')

    # Highlight the best point with a red dot
    if best_params is not None and best_score is not None:
        best_margin, best_window = best_params
        #ax.scatter(best_margin, best_window, best_score, color='red', s=50, label='Best Point', marker='o')
        #ax.text(best_margin, best_window, best_score, f'({best_margin}, {best_window}, {best_score:.2f})', color='red')

    # Labels and title
    ax.set_xlabel('Profit Margin')
    ax.set_ylabel('Future Window')
    ax.set_zlabel('F1 Score')
    ax.set_title('F1 Score vs Profit Margin and Future Window')
    
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    ax.legend()
    plt.show()

def plot_buy_signals(df, model, scaler):
    """
    Plots the historical price along with predicted buy signals from the model.
    """
    # Define the feature columns
    feature_cols = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
                    'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
                    'Bollinger_Upper', 'Bollinger_Lower', 'Stoch', 'ATR_14',
                    'CCI_20', 'Williams_%R', 'OBV', 'Momentum_10']
    
    # Check if all feature columns exist in the DataFrame
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns detected: {missing_cols}. Recalculating indicators...")
        df = add_technical_indicators(df)
    
    # Drop rows with NaN values in the feature columns
    df = df.dropna(subset=feature_cols)
    
    # Ensure there's enough data after dropping NaNs
    if df.empty:
        print("No data available after dropping NaNs. Cannot plot buy signals.")
        return

    # Prepare the features for prediction
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Make predictions using the trained model
    predictions = model.predict(X_scaled)
    
    # Extract timestamps and closing prices for plotting
    timestamps = df.index
    close_prices = df['close_price']
    
    # Identify the points where the model predicted a profitable trade (buy signal)
    buy_signals = predictions == 1

    # Plot the historical price data
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, close_prices, label='Close Price', color='blue')

    # Mark the buy signals on the plot
    plt.scatter(timestamps[buy_signals], close_prices[buy_signals],
                color='green', label='Buy Signal', marker='o', s=50)

    # Add labels, legend, and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Cryptocurrency Price with Predicted Buy Signals')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main function to run the pipeline."""
    try:
        """login()
        df = get_crypto_historical_data()
        if df is not None:
            df = create_profit_labels(df)
            # Check the label distribution
            check_label_distribution(df)
            df = add_technical_indicators(df)
            
            if not df.empty:
                train_logistic_regression(df)
        
        logout()"""
        login()
        df = get_crypto_historical_data()
        
        # Define search space using range()
        profit_margins = [x / 1000 for x in range(20, 40, 1)]
        future_windows = list(range(3, 30, 1)) 

        
        # Perform grid search
        best_params = grid_search(df, profit_margins, future_windows)
        logout()
    except Exception as e:
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        print(f"An error occurred:\n{tb_str}")

if __name__ == "__main__":
    main()