import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import imageio
import os
# Generate synthetic dataset with 3 classes
X, y = make_classification(n_samples=150, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
# Prepare model and image storage
model = LogisticRegression()
filenames = []
os.makedirs("logistic_multiclass_frames", exist_ok=True)
# Train incrementally and capture plots
for i in range(10, len(X_scaled) + 1, 10):
   model.fit(X_scaled[:i], y[:i])
   # Create mesh grid for boundary
   x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
   y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
   xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
   Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)
   plt.figure(figsize=(6, 4))
   plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
   scatter = plt.scatter(X_scaled[:i, 0], X_scaled[:i, 1], c=y[:i], cmap='viridis', edgecolor='k')
   plt.title(f'Logistic Regression Multiclass Fit ({i} samples)')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   filename = f"logistic_multiclass_frames/frame_{i}.png"
   plt.savefig(filename)
   filenames.append(filename)
   plt.close()
# Create GIF
gif_path = "logistic_regression_multiclass.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
   for filename in filenames:
       image = imageio.imread(filename)
       writer.append_data(image)
gif_path
