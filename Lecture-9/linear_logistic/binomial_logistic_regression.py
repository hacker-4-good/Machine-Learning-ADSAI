from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

X, y = load_breast_cancer(return_X_y=True)

# Apply PCA to reduce to 1 component (to match original X[:,4:5] shape)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_pca=X
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=23)
clf = LogisticRegression(max_iter=100, random_state=0)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train)) * 100
test_acc = accuracy_score(y_test, clf.predict(X_test)) * 100

print(f"Logistic Regression model training accuracy: {train_acc:.2f}%")
print(f"Logistic Regression model test accuracy: {test_acc:.2f}%")