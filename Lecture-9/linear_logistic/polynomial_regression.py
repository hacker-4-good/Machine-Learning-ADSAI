import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import imageio
import os
# 1. Generate nonlinear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])
# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Prepare directory to save frames
frame_dir = "nonlinear_frames"
os.makedirs(frame_dir, exist_ok=True)
filenames = []
# 4. Create frames using Polynomial Regression
degree = 3
for i in range(10, len(X_train)+1, 10):
   model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
   ''' 
   make_pipeline  
           ___________________________________________________
   data -> Polynomial Feature   |     LinearRegression          -> fitted model 
           _____________________|_____________________________
   '''
   model.fit(X_train[:i], y_train[:i]) # y = mx + c m and c are coefficient weights
   y_pred = model.predict(X_train[:i]) # y_pred = m(new_x) + c
   mse = mean_squared_error(y_train[:i], y_pred)
   mae = mean_absolute_error(y_train[:i], y_pred)
   r2 = r2_score(y_train[:i], y_pred)
   x_vals = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
   y_vals = model.predict(x_vals)
   plt.figure(figsize=(8, 5))
   plt.scatter(X_train[:i], y_train[:i], color='blue', label='Training data')
   plt.plot(x_vals, y_vals, color='green', linewidth=2, label='Polynomial Fit (deg=4)')
   # Residual lines
   for xi, yi, y_hat in zip(X_train[:i], y_train[:i], y_pred):
       plt.plot([xi[0], xi[0]], [yi, y_hat], color='gray', linestyle='--', linewidth=1)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title(f'Polynomial Regression Fit ({i} Samples)')
   plt.legend()
   plt.grid(True)
   fname = os.path.join(frame_dir, f"frame_{i}.png")
   plt.savefig(fname)
   filenames.append(fname)
   plt.close()
# 5. Create GIF
with imageio.get_writer("nonlinear_regression_poly4.gif", mode='I', duration=0.5) as writer:
   for filename in filenames:
       image = imageio.imread(filename)
       writer.append_data(image)
print("GIF saved as 'nonlinear_regression_poly4.gif'")

