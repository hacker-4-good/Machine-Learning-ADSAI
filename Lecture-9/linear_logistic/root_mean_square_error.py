from sklearn.metrics import root_mean_squared_error
import numpy as np
def Root_Mean_Squared_Error(Y_true, Y_pred):
 return np.sqrt(np.mean(np.subtract(Y_true,Y_pred)**2))
# Given value
# Y_true = Y (Original values)
Y_true = [1,1,2,2,4]
# Calculated values
Y_pred = [0.6, 1.29, 1.99, 2.69, 3.4]
# Calculation of RMSE using sklearn library method
print(f"RMSE Using sklearn library: {root_mean_squared_error(Y_true, Y_pred)}")
# Calculating RMSE Using custom function
print(f"RMSE Using custom function: {Root_Mean_Squared_Error(Y_true, Y_pred)}")
