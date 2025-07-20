# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import imageio
# import os
# # Sample dataset
# data = {
#    'Age': [19, 35, 26, 27, 19, 27, 27, 27, 32, 25, 25, 35, 26, 26, 26, 32, 33, 29, 47],
#    'Salary': [19000, 20000, 43000, 57000, 76000, 58000, 84000, 150000, 135000, 33000, 80000, 18000, 52000, 41000, 20000, 18000, 28000, 80000, 25000],
#    'Purchased': [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
# }
# df = pd.DataFrame(data)
# # Prepare features and scale
# X = df[['Age', 'Salary']]
# y = df['Purchased']
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # Train-test split
# X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# # Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# # Create meshgrid
# x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
# y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
#                     np.linspace(y_min, y_max, 200))
# # Prepare output directory
# os.makedirs("frames", exist_ok=True)
# filenames = []
# # Create GIF frames
# for alpha in np.linspace(0, 1, 20):  # simulate decision surface being drawn
#    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#    Z = Z.reshape(xx.shape)
#    plt.figure(figsize=(6, 4))
#    plt.contourf(xx, yy, Z, alpha=alpha, cmap='RdBu', levels=20)
#    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap='coolwarm')
#    plt.title("Logistic Regression Decision Boundary")
#    plt.xlabel("Age (scaled)")
#    plt.ylabel("Salary (scaled)")
#    filename = f"frames/frame_{int(alpha*100):03d}.png"
#    plt.savefig(filename)
#    filenames.append(filename)
#    plt.close()
# # Create GIF
# with imageio.get_writer("classification_boundary.gif", mode='I', duration=0.1) as writer:
#    for filename in filenames:
#        image = imageio.imread(filename)
#        writer.append_data(image)
# # Cleanup (optional)
# for f in filenames:
#    os.remove(f)
# print("GIF saved as 'classification_boundary.gif'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import imageio
import os

# Sample dataset
data = {
   'Age': [19, 35, 26, 27, 19, 27, 27, 27, 32, 25, 25, 35, 26, 26, 26, 32, 33, 29, 47],
   'Salary': [19000, 20000, 43000, 57000, 76000, 58000, 84000, 150000, 135000, 33000, 80000, 18000, 52000, 41000, 20000, 18000, 28000, 80000, 25000],
   'Purchased': [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

# Prepare features and scale
X = df[['Age', 'Salary']] # features
y = df['Purchased'] # target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Decision surface grid
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Get distances
decision_function = model.decision_function(X_scaled)
distances = decision_function

# Setup frame saving
os.makedirs("frames", exist_ok=True)
filenames = []

# 1. Animate decision boundary fade-in
for alpha in np.linspace(0, 1, 6):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap='RdBu', alpha=alpha, levels=[-1, 0, 1])
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Salary (scaled)")
    plt.title("Creating Decision Boundary")
    plt.colorbar(scatter)
    
    fname = f"frames/frame_boundary_{int(alpha*10)}.png"
    plt.savefig(fname, bbox_inches='tight')
    filenames.append(fname)
    plt.close()

# 2. Animate dotted error lines one by one
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.3, levels=[-1, 0, 1])
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Age (scaled)")
plt.ylabel("Salary (scaled)")
plt.title("Adding Error Distances")
plt.colorbar(scatter)

for i in range(len(X_scaled)):
    if distances[i] > 0:
        plt.arrow(X_scaled[i, 0], X_scaled[i, 1], 0, -distances[i], 
                  head_width=0.05, head_length=0.1, fc='blue', ec='blue', linestyle='--')
    else:
        plt.arrow(X_scaled[i, 0], X_scaled[i, 1], 0, -distances[i], 
                  head_width=0.05, head_length=0.1, fc='red', ec='red', linestyle='--')
    
    # Save frame after each arrow
    fname = f"frames/frame_arrow_{i:02d}.png"
    plt.savefig(fname, bbox_inches='tight')
    filenames.append(fname)

plt.close()

# Create GIF
with imageio.get_writer("animated_decision_boundary.gif", mode='I', duration=0.3) as writer:
    for f in filenames:
        image = imageio.imread(f)
        writer.append_data(image)

# Cleanup
for f in filenames:
    os.remove(f)

print("Animated GIF saved as 'animated_decision_boundary.gif'")
