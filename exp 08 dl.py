import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris Dataset
iris = load_iris()

data = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

data['Species'] = iris.target

# Features and Target
X = data.drop('Species', axis=1)
Y = data['Species']

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1
)

# KNN Model
model = KNeighborsClassifier(n_neighbors=5)

# Training
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='pink',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
