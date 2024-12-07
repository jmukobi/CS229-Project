import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the logs folder
logs_folder = "./logs"

# Combine all logs into a single DataFrame
all_logs = []
for file_name in os.listdir(logs_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(logs_folder, file_name)
        df = pd.read_csv(file_path)
        all_logs.append(df)

# Concatenate all data into a single DataFrame
data = pd.concat(all_logs, ignore_index=True)

# Ensure timestamp is in datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Sorting data by Timestamp
data = data.sort_values(by='Timestamp')

# Define gap threshold (e.g., 1 hour)
gap_threshold = pd.Timedelta(hours=10)

# Detect gaps
data['Time Difference'] = data['Timestamp'].diff()
data['New Segment'] = data['Time Difference'] > gap_threshold

# Assign segment IDs
data['Segment'] = data['New Segment'].cumsum()

# Initialize the plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Separate data into segments based on In Position and plot them
for segment_id, segment_data in data.groupby('Segment'):
    in_position_data = segment_data[segment_data['In Position'] == True]
    out_position_data = segment_data[segment_data['In Position'] == False]

    # Plot Current Price segments
    ax1.plot(
        data['Timestamp'], 
        data['Current Price'], 
        label='Current Price' if segment_id == 0 else None, 
        linewidth=2, 
        color='blue', 
        zorder=1
    )
    '''ax1.plot(
        in_position_data['Timestamp'], 
        in_position_data['Current Price'], 
        label='Current Price (In Position)' if segment_id == 0 else None, 
        linewidth=2, 
        color='red', 
        zorder=1
    )'''

# Primary y-axis settings
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Current Price")
ax1.tick_params(axis='y')

# Secondary y-axis for Buy Probability
ax2 = ax1.twinx()
ax2.plot(data['Timestamp'], data['Buy Probability'], label='Buy Probability', linewidth=2, color='orange', zorder=1)
ax2.set_ylabel("Buy Probability")
ax2.tick_params(axis='y', labelcolor='orange')

# Scatter plot for BUY Actions
buy_data = data[data['Action'] == 'BUY']
ax1.scatter(
    buy_data['Timestamp'],
    buy_data['Current Price'],
    label="BUY Action",
    color='red',
    alpha=0.7,
    edgecolor='k',
    s=50,
    zorder=2
)

# Adding titles and legends
fig.suptitle("Crypto Trading Bot Logs")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.grid()

# Show the plot
plt.tight_layout()
plt.show()
