import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_expr_and_labels

# Load dataset
X, y, expr_raw = load_expr_and_labels()
y = y["tissue"] # to predict "tissue" if it is inflamed

print("Dataset shape:", X.shape)
print("Labels:", y)

#Select 1 healthy and 1 pulpitis sample for training

healthy_idx = np.where(y == "normal pulp")[0][0]   # first healthy sample (row no) # [:2] for 2 samples
pulp_idx = np.where(y == "inflammed pulp")[0][0]      # first pulpitis sample (row no)

train_idx = np.concatenate([healthy_idx, pulp_idx]) # np.concatenate([healthy_idx, pulp_idx]) for
test_idx = [i for i in range(len(y)) if i not in train_idx]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx] # iloc because of indexing
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx] # iloc because of indexing

print(f"Training on {len(train_idx)} samples: {train_idx}")
print(f"Testing on {len(test_idx)} samples: {test_idx}")

# Training
clf = RandomForestClassifier(
    n_estimators=1,
    random_state=42
)

clf.fit(X_train, y_train)

#Evaluate on test set
y_pred = clf.predict(X_test)

# Create a DataFrame to show predictions alongside true labels
predictions = X_test.copy()
predictions['True_Label'] = y_test.values
predictions['Predicted'] = y_pred

print(predictions[['True_Label', 'Predicted']])

#primitive conf matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#visual conf matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

probs = clf.predict_proba(X_test)
for sample, true_label, prob in zip(X_test.index, y_test, probs):
    print(f"{sample}: True={true_label}, Probabilities={dict(zip(clf.classes_, prob))}, Predicted={clf.predict([X_test.loc[sample]])[0]}")

