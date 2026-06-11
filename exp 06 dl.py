import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
wine = load_wine()

data = pd.DataFrame(
    data=wine.data,
    columns=wine.feature_names
)

data['Target'] = wine.target

# Features and Target
X = data.drop('Target', axis=1)
Y = data['Target']

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
    cmap='summer',
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
