# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # Load the dataset
# data = fetch_california_housing()
# X = data.data
# y = data.target
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Create a linear regression model
# model = LinearRegression()
# # Fit the model on the training data
# model.fit(X_train, y_train)
# # Make predictions on the test set
# y_pred = model.predict(X_test)
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = mse**0.5
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)
# print("Mean Absolute Error: ", mae)
# print("Root Mean Squared Error: ", rmse) 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Load the dataset
data = fetch_california_housing()
X = data.data
y = data.target
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression model
model = LinearRegression()
# Fit the model on the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse**0.5
print('''
   Input Number -
   1 - MSE (Mean Squared Error)
   2 - R2 Score
   3 - MAE (Mean Absolute Error)
   4 - RMSE (Root Mean Squared Error)
   5 - Exit
''')
while True:
   inp = int(input("Enter the option: "))
   if inp==1:
       print("Mean Squared Error:", mse)
   if inp==2:
       print("R^2 Score:", r2)
   if inp==3:
       print("Mean Absolute Error: ", mae)
   if inp==4:
       print("Root Mean Squared Error: ", rmse)
   if inp==5:
       break
