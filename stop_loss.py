class StopLossStrategy:
    def __init__(self, threshold=0.01):
        """
        Initialize the stop loss strategy with a given threshold.
        :param threshold: The maximum loss percentage to tolerate before selling (e.g., 0.01 for 1%)
        """
        self.threshold = threshold
    
    def should_sell(self, buy_price, current_price):
        """
        Determines whether to sell based on the stop loss threshold.
        :param buy_price: The price at which the position was bought.
        :param current_price: The current price of the asset.
        :return: True if the asset should be sold, False otherwise.
        """
        # Calculate the percentage drop
        loss_percentage = (buy_price - current_price) / buy_price
        
        # Return True if the drop exceeds the threshold
        return loss_percentage >= self.threshold