from load_data import load_expr_and_labels
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
X, labels, expr_raw = load_expr_and_labels()
y = labels["pain_intensity"]

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=3
)

#train
clf = RandomForestClassifier(n_estimators=9, random_state=3)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=y.unique())
cm_df = pd.DataFrame(
    cm,
    index=[f"True: {c}" for c in y.unique()],
    columns=[f"Pred: {c}" for c in y.unique()]
)

print("\nConfusion Matrix:")
print(cm_df)

#Confusion matrix visual
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.show()

#Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Predictions per patient
results = pd.DataFrame({
    "True Label": y_test,
    "Predicted Label": y_pred
}, index=y_test.index)
results["Correct?"] = results["True Label"] == results["Predicted Label"]

print("\nPrediction results per patient:")
print(results)

#Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop 20 important features (genes):")
print(importances_sorted.head(20))

#plot top 20 features
plt.figure(figsize=(10,6))
sns.barplot(x=importances_sorted.head(20).values, y=importances_sorted.head(20).index)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Gene Probe")
plt.show()
