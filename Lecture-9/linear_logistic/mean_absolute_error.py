from sklearn.metrics import mean_absolute_error
import numpy as np
def Mean_Absolute_Error(Y_true, Y_pred):
 return  np.abs(np.subtract(Y_true, Y_pred)).mean()
# Given value
# Y_true = Y (Original values)
Y_true = [1,1,2,2,4]
# Calculated values
Y_pred = [0.6, 1.29, 1.99, 2.69, 3.4]
# Calculation of MAE using sklearn library method
print(f"MSE Using sklearn library: {mean_absolute_error(Y_true, Y_pred)}")
# Calculating MAE Using custom function
print(f"MSE Using custom function: {Mean_Absolute_Error(Y_true, Y_pred)}")
