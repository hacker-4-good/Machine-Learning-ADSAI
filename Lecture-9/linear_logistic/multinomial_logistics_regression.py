from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
''' Output -
Logistic Regression model accuracy: 96.66%
'''
