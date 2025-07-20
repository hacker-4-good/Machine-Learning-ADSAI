from sklearn.metrics import r2_score
# Given value
# Y_true = Y (Original values)
Y_true = [1,1,2,2,4]
# Calculated values
Y_pred = [0.6, 1.29, 1.99, 2.69, 3.4]
# Calculation of R2-score using sklearn library method
print(f"R2-Score Using sklearn library: {r2_score(Y_true, Y_pred)}")
'''R2-Score Using sklearn library: 0.81995'''
