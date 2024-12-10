import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the sigmoid function
rsi_14_param = -1.46
rsi_28_param = 1.17

# Sigmoid function
def sigmoid(x, y, a, b):
    return 1 / (1 + np.exp(-(a * x + b * y)))

# Generate data
x = np.linspace(0, 3, 100)  # RSI-14 values
y = np.linspace(0, 3, 100)  # RSI-28 values
X, Y = np.meshgrid(x, y)
Z = sigmoid(X, Y, rsi_14_param, rsi_28_param)

# Plot the 3D sigmoid
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# Labels and title
ax.set_xlabel('14-hour RSI')
ax.set_ylabel('28-hour RSI')
ax.set_zlabel('Sigmoid Output')
ax.set_title('3D Sigmoid Function')

plt.show()
