# CS229 Project: Cryptocurrency Trading Bot

This repository contains the code and resources for a cryptocurrency trading bot developed as part of my CS229 project. The bot uses logistic regression to predict price movements and automates trading decisions using the Robinhood API.

---

## Directory Structure

### Root Files
- **`training.py`**: Script for training the logistic regression model on historical cryptocurrency data.
- **`logreg_trader.py`**: Real-time trading bot that uses the trained logistic regression model to make buy/hold/sell decisions.

---

### Folders

#### `logs/`
Contains CSV files recording logs of trading activity during live testing. Each file is timestamped with the start time of the script. Logs include details such as:
- Timestamps of actions
- Buy/Hold/Sell decisions
- Model parameters and weights
- Current price and profit margins
- Whether the bot is currently in a position

#### `plots/`
Contains visualizations of the training and trading processes, including:
- **`buy_signals.png`**: A plot showing historical price data with marked buy signals.
- **`profit_margin_vs_window.png`**: A 3D plot illustrating the F1 score for various profit margins and future windows during grid search.
- **`trading_data.png`**: A plot of historical price data used for training.

---

### Other Files

- **`.gitignore`**: Specifies files and folders to exclude from version control, such as logs, configuration, and cache files.
- **`configuration.py`**: Stores Robinhood credentials (username and password). *Ensure this file is not committed to the repository.*
- **`requirements.txt`**: Contains all necessary dependencies for the project. Install them with:
  ```bash
  pip install -r requirements.txt
  ```
- **`plotting.py`**: Plots logged live training data.
- **`README.md`**: This file, providing an overview of the project and directory contents.


---

## Getting Started

### Prerequisites
- Python 3.8+
- A Robinhood account
- Dependencies from `requirements.txt`

### Running the Scripts
1. **Training**: Run `training.py` to train the model on historical data.
   ```bash
   python training.py
   ```
2. **Live Trading**: Use `logreg_trader.py` to run the real-time trading bot.
   ```bash
   python logreg_trader.py
   ```

---

## Notes
- Ensure your Robinhood credentials are stored securely in `configuration.py` and not shared or committed.
- Logs and plots generated during execution are stored in their respective folders for analysis.

---

Feel free to modify the project or open issues if you have questions!