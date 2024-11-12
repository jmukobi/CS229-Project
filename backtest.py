import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import get_crypto_historical_data, create_profit_labels, add_technical_indicators, train_model, make_prediction, login, logout
from stop_loss import StopLossStrategy
from tqdm import tqdm

# Constants
SYMBOL = "ETH"
INITIAL_CAPITAL = 10000  # Starting with $10,000 for the backtest
TRADE_SIZE = 1000       # Each trade will buy/sell $1,000 worth
PROFIT_MARGIN = 0.01    # Profit target (1% above buy price)
FUTURE_WINDOW = 24
STOP_LOSS_THRESHOLD = 0.01  # 1% stop loss

def backtest():
    # Step 1: Log in to Robinhood
    login()

    # Step 2: Get historical data
    df = get_crypto_historical_data(symbol=SYMBOL)
    if df is None:
        print("Failed to retrieve historical data.")
        logout()
        return

    # Step 3: Create profit labels and technical indicators
    df = create_profit_labels(df, profit_margin=PROFIT_MARGIN, future_window=FUTURE_WINDOW)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)

    # Initialize the stop loss strategy
    stop_loss_strategy = StopLossStrategy(threshold=STOP_LOSS_THRESHOLD)

    # Set up variables for tracking performance
    capital = INITIAL_CAPITAL
    positions = 0  # Track open positions
    buy_price = None  # Track the buy price of the current position
    target_sell_price = None  # Target sell price for profit margin
    trade_log = []  # Record each trade

    # Step 4: Run backtest
    for i in tqdm(range(FUTURE_WINDOW, len(df)), desc="Backtesting Progress"):
        # Use all data up to the current point for training
        df_train = df.iloc[:i]
        model, scaler = train_model(df_train)

        # If the model could not be trained (only one class was present), skip to the next iteration
        if model is None or scaler is None:
            print("Training skipped due to insufficient class variety.")
            continue

        # Test the next window period to simulate real trading
        df_test = df.iloc[i:i+FUTURE_WINDOW]

        predictions = make_prediction(df_test, model, scaler)
        if predictions is None:
            print("Skipping current backtest window due to insufficient data.")
            continue  # Skip to the next iteration if predictions could not be made

        # Step 5: Execute trades based on predictions and stop loss strategy
        for j, prediction in enumerate(predictions):
            price = df_test['close_price'].iloc[j]

            # Buy logic
            if prediction == 1 and capital >= TRADE_SIZE:
                if positions == 0:  # Only buy if no open position
                    buy_price = price
                    target_sell_price = buy_price * (1 + PROFIT_MARGIN)
                    positions += TRADE_SIZE / price
                    capital -= TRADE_SIZE
                    trade_log.append({"Action": "BUY", "Price": price, "Capital": capital, "Positions": positions})

            # Sell logic if we have an open position
            if positions > 0:
                # Check if we hit the profit target
                if price >= target_sell_price:
                    capital += positions * price
                    positions = 0
                    trade_log.append({"Action": "SELL (PROFIT TARGET)", "Price": price, "Capital": capital, "Positions": positions})
                    buy_price = None
                    target_sell_price = None

                # Check stop loss condition
                elif stop_loss_strategy.should_sell(buy_price, price):
                    capital += positions * price
                    positions = 0
                    trade_log.append({"Action": "SELL (STOP LOSS)", "Price": price, "Capital": capital, "Positions": positions})
                    buy_price = None
                    target_sell_price = None

    # Final evaluation of remaining positions
    if positions > 0:
        final_price = df['close_price'].iloc[-1]
        capital += positions * final_price
        positions = 0
        trade_log.append({"Action": "SELL (FINAL)", "Price": final_price, "Capital": capital, "Positions": positions})

    # Step 6: Log out from Robinhood
    logout()

    # Step 7: Analyze results
    df_trades = pd.DataFrame(trade_log)
    print(f"\nInitial Capital: ${INITIAL_CAPITAL}")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Profit/Loss: ${capital - INITIAL_CAPITAL:.2f}")

    # Plotting trade actions on price data
    plt.figure(figsize=(12, 6))
    if not df_test.empty:
        # Ensure the DataFrame has the 'timestamp' column
        if 'timestamp' not in df_test.columns:
            df_test.reset_index(inplace=True)
        
        plt.plot(df_test['timestamp'], df_test['close_price'], label='Close Price', color='blue')
    buy_signals = df_trades[df_trades["Action"].str.contains("BUY")]
    profit_sell_signals = df_trades[df_trades["Action"].str.contains("PROFIT TARGET")]
    stop_loss_signals = df_trades[df_trades["Action"].str.contains("STOP LOSS")]
    
    # Plot buy and sell signals
    plt.scatter(buy_signals.index, buy_signals["Price"], color='green', marker='^', label='Buy Signal')
    plt.scatter(profit_sell_signals.index, profit_sell_signals["Price"], color='orange', marker='o', label='Sell (Profit Target)')
    plt.scatter(stop_loss_signals.index, stop_loss_signals["Price"], color='red', marker='v', label='Sell (Stop Loss)')

    # Add labels, legend, and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Backtesting Results for {SYMBOL}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    backtest()
