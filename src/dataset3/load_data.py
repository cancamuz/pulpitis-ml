import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# --- 1. Load data ---
data_path = r"C:\Users\ccz\Documents\pulpitis_ml\data\metadata.csv"
df = pd.read_csv(data_path)

# Define features (X) and target (y)
X = df.drop(columns=["Sample ID", "Group", "Cold pulp test"])
y = df["Group"]

# --- 2. Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Identify categorical & numerical columns ---
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# --- 4. Preprocessing ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# Fit the preprocessor on the training data only
preprocessor.fit(X_train)

# Get encoded feature names
encoded_features = preprocessor.get_feature_names_out()

print("Features used for training:")
print(encoded_features)
# --- 5. Define models and hyperparameter grids ---

# Logistic Regression
log_reg = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])
log_params = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__solver": ["liblinear", "lbfgs"]
}

# Random Forest
rf = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])
rf_params = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5, 10]
}

# Neural Network
mlp = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", MLPClassifier(max_iter=1000, random_state=42))
])
mlp_params = {
    "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "clf__activation": ["relu", "tanh"],
    "clf__alpha": [0.0001, 0.001, 0.01]
}

# --- 6. Run Grid Search ---
models = {
    "Logistic Regression": (log_reg, log_params),
    "Random Forest": (rf, rf_params),
    "Neural Network": (mlp, mlp_params)
}

best_models = {}

for name, (pipe, params) in models.items():
    grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)
    print(f"\n{name} Best Params: {grid.best_params_}")
    print(f"{name} Best CV Accuracy: {grid.best_score_:.3f}")

    y_pred = grid.predict(X_test)
    print(f"{name} Test Performance:\n", classification_report(y_test, y_pred))

    best_models[name] = grid.best_estimator_

