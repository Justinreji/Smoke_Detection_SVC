import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)

# Load and clean the dataset
df = pd.read_csv("smoke_detection_iot.csv")
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')
df.drop(['Unnamed: 0', 'UTC'], axis=1, inplace=True)
df.dropna(inplace=True)

# Bar plot for target class balance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Fire Alarm")
plt.title("Class Distribution of Fire Alarm")
plt.xlabel("Fire Alarm (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Sample and normalize the data
df_sample = df.sample(n=10000, random_state=42)
X = df_sample.iloc[:, :-1].to_numpy()
y = df_sample.iloc[:, -1].to_numpy()
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary to store results
model_scores = {}

# -----------------------------
# Model 1: SVC with Linear Kernel
# -----------------------------
print("\n--- Linear Kernel SVC ---")
param_grid_linear = {"C": np.linspace(0.9, 1.0, num=20)}
grid_linear = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid_linear, cv=5, scoring="accuracy")
grid_linear.fit(X_train, y_train)

linear_best = grid_linear.best_estimator_
linear_best.fit(X_train, y_train)
y_pred_linear = linear_best.predict(X_test)

print(f"Train Score: {linear_best.score(X_train, y_train):.3f}")
print(f"Test Score: {linear_best.score(X_test, y_test):.3f}")
print("\nClassification Report (Linear SVM):")
print(classification_report(y_test, y_pred_linear))

cm_linear = confusion_matrix(y_test, y_pred_linear)
ConfusionMatrixDisplay(cm_linear, display_labels=linear_best.classes_).plot()
plt.title("Confusion Matrix: Linear SVC")
plt.show()

# Save metrics
model_scores['Linear SVM'] = {
    "Accuracy": accuracy_score(y_test, y_pred_linear),
    "Precision": precision_score(y_test, y_pred_linear),
    "Recall": recall_score(y_test, y_pred_linear),
    "F1 Score": f1_score(y_test, y_pred_linear)
}

# -----------------------------
# Model 2: SVC with RBF Kernel
# -----------------------------
print("\n--- RBF Kernel SVC ---")
param_grid_rbf = {
    "C": np.linspace(0.5, 1.0, num=5),
    "gamma": np.linspace(0.01, 1.0, num=4)
}
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid_rbf, cv=5, scoring="accuracy")
grid_rbf.fit(X_train, y_train)

rbf_best = grid_rbf.best_estimator_
rbf_best.fit(X_train, y_train)
y_pred_rbf = rbf_best.predict(X_test)

print(f"Train Score: {rbf_best.score(X_train, y_train):.3f}")
print(f"Test Score: {rbf_best.score(X_test, y_test):.3f}")
print("\nClassification Report (RBF SVM):")
print(classification_report(y_test, y_pred_rbf))

cm_rbf = confusion_matrix(y_test, y_pred_rbf)
ConfusionMatrixDisplay(cm_rbf, display_labels=rbf_best.classes_).plot()
plt.title("Confusion Matrix: RBF SVC")
plt.show()

# Save metrics
model_scores['RBF SVM'] = {
    "Accuracy": accuracy_score(y_test, y_pred_rbf),
    "Precision": precision_score(y_test, y_pred_rbf),
    "Recall": recall_score(y_test, y_pred_rbf),
    "F1 Score": f1_score(y_test, y_pred_rbf)
}

# -----------------------------
# Comparison Table
# -----------------------------
comparison_df = pd.DataFrame(model_scores).T
print("\n--- Model Comparison ---")
print(comparison_df)