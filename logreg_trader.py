import numpy as np
import robin_stocks.robinhood as r
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from datetime import datetime
import time
import csv
from configuration import username, password
import traceback

# Constants
SYMBOL = "ETH"
DATA_INTERVAL = "hour"
DATA_SPAN = "month"
STOP_LOSS_THRESHOLD = 0.05
POSITION_SIZE = 0.1
INTERVAL = 10  # seconds between each decision
SELL_SAFETEY_FACTOR = 1
BUY_PROBABILITY_THRESHOLD = 0.75

# Technical indicators to use
TECHNICAL_INDICATORS = [
    'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50',
    'RSI_14', 'RSI_28', 'MACD', 'MACD_Signal',
    'Bollinger_Upper', 'Bollinger_Lower', 'Stoch', 'ATR_14',
    'CCI_20', 'Williams_%R', 'OBV', 'Momentum_10'
]

# Global Variables
in_position_flag = False
buy_price = 0.01
log_file = None
purchasing_power = 0.0

# Login to Robinhood
def login_to_robinhood():
    try:
        r.login(username, password, expiresIn=86400000, by_sms=True)
        print("Logged in to Robinhood")
    except Exception as e:
        print("Failed to log in:", e)
        exit()

# Logout
def logout_from_robinhood():
    r.logout()
    print("Logged out of Robinhood")

# Create log file
def create_csv_log():
    global log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"trading_log_{timestamp}.csv"
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp", "Action", "Parameters", "Model Weights", "Buy Price", "Current Price", "Profit Margin", "In Position", "Buy Probability"
        ])
    print(f"Log file created: {log_file}")

# Fetch crypto data
def fetch_crypto_data():
    try:
        data = r.crypto.get_crypto_historicals(SYMBOL, interval=DATA_INTERVAL, span=DATA_SPAN, bounds='24_7')
        if not data:
            print(f"No data found for {SYMBOL}")
            return None
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['begins_at'])
        df.set_index('timestamp', inplace=True)
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce')
        df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce')
        df['open_price'] = pd.to_numeric(df['open_price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        return df.dropna()
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Add technical indicators
def add_technical_indicators(df):
    df['SMA_10'] = ta.sma(df['close_price'], length=10)
    df['SMA_50'] = ta.sma(df['close_price'], length=50)
    df['EMA_10'] = ta.ema(df['close_price'], length=10)
    df['EMA_50'] = ta.ema(df['close_price'], length=50)
    df['RSI_14'] = ta.rsi(df['close_price'], length=14)
    df['RSI_28'] = ta.rsi(df['close_price'], length=28)
    macd = ta.macd(df['close_price'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    bbands = ta.bbands(df['close_price'], length=20)
    df['Bollinger_Upper'] = bbands['BBU_20_2.0']
    df['Bollinger_Lower'] = bbands['BBL_20_2.0']
    stoch = ta.stoch(df['high_price'], df['low_price'], df['close_price'])
    df['Stoch'] = stoch['STOCHk_14_3_3']
    df['ATR_14'] = ta.atr(df['high_price'], df['low_price'], df['close_price'], length=14)
    df['CCI_20'] = ta.cci(df['high_price'], df['low_price'], df['close_price'], length=20)
    df['Williams_%R'] = ta.willr(df['high_price'], df['low_price'], df['close_price'], length=14)
    df['OBV'] = ta.obv(df['close_price'], df['volume'])
    df['Momentum_10'] = ta.mom(df['close_price'], length=10)
    return df.dropna()

# Run grid search
def run_grid_search(df):
    best_params = None
    best_score = 0
    for margin in [x / 1000 for x in range(20, 40, 1)]:
        for window in range(2, 40, 1):
            try:
                # Step 1: Label the data
                labeled_df = create_profit_labels(df.copy(), margin, window)
                labeled_df = add_technical_indicators(labeled_df)
                
                if labeled_df.empty:
                    continue  # Skip if no data after labeling

                # Step 2: Prepare features and labels
                X = labeled_df[TECHNICAL_INDICATORS]
                y = labeled_df['label']

                # Skip if only one class is present
                if len(y.unique()) < 2:
                    #print(f"Skipping margin={margin}, window={window}: only one class present (labels={y.unique()})")
                    continue

                # Step 3: Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Step 4: Train the model
                scaler = StandardScaler().fit(X_train)
                model = LogisticRegression().fit(scaler.transform(X_train), y_train)

                # Step 5: Evaluate the model
                score = f1_score(y_test, model.predict(scaler.transform(X_test)), zero_division=1)

                # Step 6: Update best parameters if this one is better
                if score > best_score:
                    best_score = score
                    best_params = (margin, window)
            except Exception as e:
                # Handle any error during grid search
                #print(f"Skipping margin={margin}, window={window} due to error: {e}")
                continue
        #print(f"Best score: {best_score}, Best params: {best_params}")
    return best_params, best_score


# Create profit labels
def create_profit_labels(df, profit_margin, future_window):
    df['future_price'] = df['close_price'].shift(-future_window)
    df['profit_threshold'] = df['close_price'] * (1 + profit_margin)
    df['label'] = (df['future_price'] > df['profit_threshold']).astype(int)
    return df.dropna()

# Train model
def train_model(df, best_params):
    labeled_df = create_profit_labels(df.copy(), *best_params)
    labeled_df = add_technical_indicators(labeled_df)
    X = labeled_df[TECHNICAL_INDICATORS]
    y = labeled_df['label']
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    return model, scaler

# Execute buy
def execute_buy(current_price, sell_price, buy_quantity):
    global in_position_flag, buy_price
    print(f"Buying at {current_price}, quantity: {buy_quantity}, sell price: {sell_price}")
    #return
    try:
        print(r.orders.order_buy_crypto_by_price(SYMBOL, buy_quantity)) 
        in_position_flag = True
        buy_price = current_price
        print(f"Bought at {current_price}")
        time.sleep(20)  # Wait for order to be filled
        #get quantity held
        quantity = r.crypto.get_crypto_positions()[0]['quantity_available']
        print(f"Quantity held: {quantity}")
        print(r.orders.order_sell_crypto_limit(SYMBOL, quantity, sell_price))
        print(f"Limit sell order placed at {sell_price}.")

    except Exception as e:
        print("Failed to buy or place limit sell:", e)

# Check stop loss
def check_stop_loss(current_price, buy_price):
    #global buy_price
    print(f"Checking stop loss: Buy price: {buy_price}, Current price: {current_price}")
    time.sleep(5)
    return (buy_price - current_price) / buy_price >= STOP_LOSS_THRESHOLD

# Sell position
def sell_position():
    global in_position_flag

    print("Selling position.")
    #return
    try:
        print(r.orders.order_sell_crypto_by_quantity(SYMBOL, r.crypto.get_crypto_positions()[0]['quantity']))
        in_position_flag = False
        print("Position sold.")
    except Exception as e:
        print("Failed to sell position:", e)

def log_action(action, parameters, model_weights, current_price, profit_margin, buy_probability=None):
    global in_position_flag, buy_price

    # Serialize weights into a compact format
    try:
        weights_string = np.array2string(model_weights, separator=" ", suppress_small=True).replace("\n", " ")
    except:
        weights_string = ""
    #print(model_weights)
    #print(weights_string)
    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action,
            parameters,
            weights_string,
            buy_price,
            current_price,
            profit_margin,
            in_position_flag,
            buy_probability
        ])


def get_buy_probability(model, scaler, df):
    """
    Get the probability of a 'buy' signal from the logistic regression model.

    :param model: Trained logistic regression model.
    :param scaler: StandardScaler object for normalizing the data.
    :param df: DataFrame containing the latest technical indicators.
    :return: Probability of 'buy' class (class 1).
    """
    # Transform the latest data point
    transformed_data = scaler.transform(df[TECHNICAL_INDICATORS].tail(1))

    # Get probabilities for each class
    probabilities = model.predict_proba(transformed_data)

    # Return probability of class 1 ('buy')
    return probabilities[0][1]


# Main function
def main():
    login_to_robinhood()
    create_csv_log()
    try:
        while True:
            df = fetch_crypto_data()
            
            # Check if in position by querying held crypto
            global in_position_flag, buy_price
            crypto_positions = r.crypto.get_crypto_positions()
            in_position_flag = False  # Reset flag initially

            for position in crypto_positions:
                # Safely access keys using .get()
                currency_code = position.get('currency', {}).get('code')
                quantity_available = position.get('quantity', 0)

                if currency_code == SYMBOL and float(quantity_available) > 0:
                    in_position_flag = True
                    #buy_price = float(position['cost_bases'][0]['direct_cost_basis']) / float(position['quantity_available'])
                    print(f"In position: Holding {position['quantity']} {SYMBOL} at an average buy price of {buy_price:.2f}")
                    break

            if not df.empty:
                current_price = df['close_price'].iloc[-1]
                if in_position_flag:
                    print(f"buy price: {buy_price}, current price: {current_price}")
                    if check_stop_loss(current_price, buy_price):
                        sell_position()
                        print("Stop loss triggered.")
                        log_action("STOP_LOSS", "", "", current_price, 0)
                        continue

                purchasing_power = float(r.profiles.load_account_profile()['crypto_buying_power'])
                print(f"Purchasing power: {purchasing_power}")

                best_params, best_score = run_grid_search(df)
                print(f"Best parameters: {best_params}, Best score: {best_score:.2f}")
                print(f"Current price: {current_price}")

                if best_params:
                    model, scaler = train_model(df, best_params)
                    df = add_technical_indicators(df)
                    buy_probability = get_buy_probability(model, scaler, df)

                    print(f"Buy probability: {buy_probability:.2f}")
                    if not in_position_flag:
                        if buy_probability >= BUY_PROBABILITY_THRESHOLD:  # High-confidence buy threshold

                            sell_price = round(current_price * (1 + best_params[0]/SELL_SAFETEY_FACTOR), 2)
                            buy_quantity = purchasing_power*POSITION_SIZE
                            execute_buy(current_price, sell_price, buy_quantity)
                            log_action("BUY", best_params, model.coef_, current_price, best_params[0], buy_probability)
                        else:
                            log_action("HOLD", best_params, model.coef_, current_price, best_params[0], buy_probability)
                    else:
                        log_action("HOLD", best_params, model.coef_, current_price, best_params[0], buy_probability)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        logout_from_robinhood()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        logout_from_robinhood()

if __name__ == "__main__":
    #login_to_robinhood()

    #sell_position()
    while True:
        try:
            main()
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            logout_from_robinhood()
            time.sleep(10)
            continue
    #main()
