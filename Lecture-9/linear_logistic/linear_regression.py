import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import imageio
import os
# Dataset from the image
data = {
   'Temperature': np.linspace(10, 60, 50),
   'Wind Speed': [5 + np.sin(i / 5) * 10 + i for i in range(50)]
}
df = pd.DataFrame(data)
X = df[['Temperature']]
y = df['Wind Speed']
# Define x range for predictions
x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# Prepare output folder
os.makedirs("regression_gif_frames", exist_ok=True)
filenames = []
# Fit model incrementally (simulated for animation)
for i in range(2, len(X) + 1): # if this for loop runs 50 rows
   model = LinearRegression()
   model.fit(X[:i], y[:i]) # i = 2  X[0:2] 2 values y = mx + c 
   y_pred = model.predict(x_vals)

   # Plot
   plt.figure(figsize=(6, 4))
   plt.scatter(X, y, color='blue', label='Data')
   plt.plot(x_vals, y_pred, color='red', label='Regression Line')
   plt.title(f'Regression Line Fitting - Step {i}')
   plt.xlabel('Temperature')
   plt.ylabel('Wind Speed')
   plt.legend()
   plt.grid(True)

   filename = f"regression_gif_frames/frame_{i:02d}.png"
   plt.savefig(filename)
   filenames.append(filename)
   plt.close()
# Create GIF
gif_path = "regression_fitting.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.4) as writer:
   for filename in filenames:
       image = imageio.imread(filename)
       writer.append_data(image)
# Cleanup frames
for f in filenames:
   os.remove(f)
gif_path
