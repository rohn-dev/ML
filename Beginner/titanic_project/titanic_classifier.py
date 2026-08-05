"""
=============================================================================
TITANIC SURVIVAL CLASSIFIER
=============================================================================
A complete, beginner-friendly, end-to-end machine learning pipeline that:
  1. Loads and explores the Titanic dataset (train.csv / test.csv).
  2. Cleans data and engineers new features.
  3. Trains and compares six classification algorithms.
  4. Tunes the best model with GridSearchCV.
  5. Evaluates the tuned model (metrics, confusion matrix, ROC curve,
     feature importance, learning curve, 5-fold cross-validation).
  6. Generates `submission.csv` for the Kaggle-style test set.
  7. Persists the trained pipeline with joblib.

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib

Usage:
    Place `train.csv` and `test.csv` (standard Kaggle Titanic format) in the
    same folder as this script, then run:

        python titanic_classifier.py

Outputs (written to ./outputs/):
    - submission.csv
    - model_comparison.csv
    - confusion_matrix.png
    - roc_curve.png
    - feature_importance.png
    - learning_curve.png
    - best_titanic_model.joblib
=============================================================================
"""

# =============================================================================
# SECTION 0: IMPORTS
# =============================================================================
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")  # keep console output readable for beginners

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Consistent, pleasant plot styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# All output artifacts (plots, csv, model) go into ./outputs/
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Helper to save a matplotlib figure to the outputs folder and show it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> saved plot: {path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("=" * 70)
print("SECTION 1: LOADING DATA")
print("=" * 70)

# The classic Kaggle Titanic files are expected in the working directory.
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")
print("\nMissing values in train set:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])

# Keep the test PassengerIds for the final submission file before we touch
# anything else.
test_passenger_ids = test_df["PassengerId"].copy()


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================
# We engineer features on train and test together (concatenated) so that both
# sets end up with identical categories/encodings, then split them apart
# again. The target column ("Survived") is excluded from the concatenation.
print("\n" + "=" * 70)
print("SECTION 2: FEATURE ENGINEERING")
print("=" * 70)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new, more predictive features from the raw Titanic columns.

    New features created:
      - Title       : Extracted from the passenger's Name (Mr, Mrs, Miss...).
      - FamilySize  : SibSp + Parch + 1 (the passenger themself).
      - IsAlone     : 1 if the passenger has no family aboard, else 0.
      - Deck        : First letter of the Cabin code (M for missing/unknown).
      - FarePerPerson : Fare divided by FamilySize (accounts for shared tickets).
    """
    df = df.copy()

    # --- Title: a strong proxy for age, sex, and social status ---
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    # Group rare/foreign titles into broader, more populated buckets
    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # --- Family features ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Deck: first letter of the cabin code; "M" = missing ---
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("M")

    # --- Fare per person (helps normalize group-ticket fares) ---
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df


train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

print("New engineered columns: Title, FamilySize, IsAlone, Deck, FarePerPerson")
print("\nTitle distribution (train):")
print(train_fe["Title"].value_counts())


# =============================================================================
# SECTION 3: DROP IRRELEVANT COLUMNS
# =============================================================================
# PassengerId, Name, Ticket, and Cabin are either unique identifiers or too
# sparse/high-cardinality to use directly (we already extracted the useful
# signal from Name -> Title and Cabin -> Deck).
print("\n" + "=" * 70)
print("SECTION 3: DROPPING IRRELEVANT COLUMNS")
print("=" * 70)

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]
train_fe = train_fe.drop(columns=DROP_COLS)
test_fe = test_fe.drop(columns=DROP_COLS)
print(f"Dropped columns: {DROP_COLS}")

# Separate features (X) and target (y) for the training data
y = train_fe["Survived"]
X = train_fe.drop(columns=["Survived"])
X_test_final = test_fe.copy()

print(f"\nFinal feature columns: {list(X.columns)}")


# =============================================================================
# SECTION 4: PREPROCESSING PIPELINE (missing values + encoding + scaling)
# =============================================================================
# We use a ColumnTransformer so that numeric and categorical columns get
# different treatment, and wrap everything in a scikit-learn Pipeline so the
# exact same preprocessing (fit on training data only) is re-applied to the
# validation set and the final test set without leakage.
print("\n" + "=" * 70)
print("SECTION 4: BUILDING PREPROCESSING PIPELINE")
print("=" * 70)

numeric_features = ["Age", "Fare", "FarePerPerson", "FamilySize", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]

# Numeric pipeline: fill missing values with the median, then scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline: fill missing values with the most frequent value,
# then one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

print("Numeric features    :", numeric_features)
print("Categorical features:", categorical_features)


# =============================================================================
# SECTION 5: TRAIN / VALIDATION SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set  : {X_train.shape[0]} rows")
print(f"Validation set: {X_val.shape[0]} rows")


# =============================================================================
# SECTION 6: TRAIN AND COMPARE MULTIPLE MODELS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: TRAINING AND COMPARING MODELS")
print("=" * 70)

# Each model is wrapped in the SAME preprocessing pipeline, so comparisons
# are fair and no manual re-encoding is needed.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]  # all six models support predict_proba

    # 5-fold cross-validation on the training set (bonus requirement)
    cv_scores = cross_val_score(
        pipe, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1-score": f1_score(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_proba),
        "CV Accuracy (mean)": cv_scores.mean(),
        "CV Accuracy (std)": cv_scores.std(),
    })
    print(f"Trained: {name:<25} | Val Accuracy: {results[-1]['Accuracy']:.4f} "
          f"| CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False).reset_index(drop=True)

print("\n" + "-" * 70)
print("MODEL COMPARISON TABLE (sorted by F1-score)")
print("-" * 70)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(f"\n  -> saved table: {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")


# =============================================================================
# SECTION 7: SELECT THE BEST MODEL
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: SELECTING THE BEST MODEL")
print("=" * 70)

best_model_name = results_df.iloc[0]["Model"]
print(f"Best-performing model (by F1-score on validation set): {best_model_name}")


# =============================================================================
# SECTION 8: HYPERPARAMETER TUNING WITH GridSearchCV
# =============================================================================
print("\n" + "=" * 70)
print(f"SECTION 8: TUNING '{best_model_name}' WITH GridSearchCV")
print("=" * 70)

# A focused hyperparameter grid for each possible best model. Only the grid
# for the actually-selected model is used, keeping the search fast.
param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "Support Vector Machine": {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto"],
        "classifier__kernel": ["rbf", "linear"],
    },
    "K-Nearest Neighbors": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
    },
}

base_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name]),
])

grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grids[best_model_name],
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validated F1-score (train folds): {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_


# =============================================================================
# SECTION 9: EVALUATE THE FINAL (TUNED) MODEL ON THE VALIDATION SET
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: EVALUATING THE FINAL MODEL ON THE VALIDATION SET")
print("=" * 70)

y_val_pred = final_model.predict(X_val)
y_val_proba = final_model.predict_proba(X_val)[:, 1]

final_accuracy = accuracy_score(y_val, y_val_pred)
final_precision = precision_score(y_val, y_val_pred)
final_recall = recall_score(y_val, y_val_pred)
final_f1 = f1_score(y_val, y_val_pred)
final_roc_auc = roc_auc_score(y_val, y_val_proba)

print(f"Final Accuracy : {final_accuracy:.4f}")
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall   : {final_recall:.4f}")
print(f"Final F1-score : {final_f1:.4f}")
print(f"Final ROC-AUC  : {final_roc_auc:.4f}")

# 5-fold cross-validation score of the FINAL tuned model (bonus requirement)
final_cv_scores = cross_val_score(
    final_model, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy",
)
print(f"Final 5-fold CV Accuracy: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std():.4f})")

# --- Confusion Matrix plot ---
cm = confusion_matrix(y_val, y_val_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Did not survive", "Survived"],
    yticklabels=["Did not survive", "Survived"], ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name} (tuned)")
save_and_show(fig, "confusion_matrix.png")

# --- ROC Curve plot ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))
RocCurveDisplay.from_predictions(y_val, y_val_proba, ax=ax, name=best_model_name)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_title(f"ROC Curve — {best_model_name} (tuned) | AUC = {final_roc_auc:.3f}")
ax.legend()
save_and_show(fig, "roc_curve.png")


# =============================================================================
# SECTION 10: FEATURE IMPORTANCE (tree-based models)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: FEATURE IMPORTANCE")
print("=" * 70)

classifier_step = final_model.named_steps["classifier"]

# Helper to pull human-readable feature names out of a fitted preprocessor
def get_feature_names(fitted_preprocessor):
    ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    return np.concatenate([numeric_features, cat_names])


def plot_feature_importance(importances, feature_names, model_label):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature",
                palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Top 15 Feature Importances — {model_label}")
    save_and_show(fig, "feature_importance.png")
    print(importance_df.to_string(index=False))


if hasattr(classifier_step, "feature_importances_"):
    # The tuned/best model is itself tree-based -> use it directly.
    names = get_feature_names(final_model.named_steps["preprocessor"])
    plot_feature_importance(classifier_step.feature_importances_, names, f"{best_model_name} (tuned)")
else:
    # The best model (e.g. Logistic Regression / SVM / KNN) has no native
    # feature_importances_. To still satisfy the "feature importance for
    # tree-based models" requirement, we fit a supplementary Random Forest
    # on the same training data purely for interpretability purposes. This
    # does NOT replace the selected best model used for predictions.
    print(f"'{best_model_name}' has no feature_importances_ attribute "
          f"(only tree-based models expose this). Fitting a supplementary "
          f"Random Forest on the training data for interpretability only.")
    rf_explainer = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
    ])
    rf_explainer.fit(X_train, y_train)
    names = get_feature_names(rf_explainer.named_steps["preprocessor"])
    plot_feature_importance(
        rf_explainer.named_steps["classifier"].feature_importances_,
        names,
        "Random Forest (supplementary, for interpretability)",
    )


# =============================================================================
# SECTION 11: LEARNING CURVE (bonus)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: LEARNING CURVE")
print("=" * 70)

train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
)

train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_sizes, train_mean, "o-", color="tab:blue", label="Training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="tab:blue")
ax.plot(train_sizes, val_mean, "o-", color="tab:orange", label="Cross-validation score")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="tab:orange")
ax.set_xlabel("Training examples")
ax.set_ylabel("Accuracy")
ax.set_title(f"Learning Curve — {best_model_name} (tuned)")
ax.legend(loc="best")
save_and_show(fig, "learning_curve.png")


# =============================================================================
# SECTION 12: GENERATE PREDICTIONS FOR test.csv AND SAVE submission.csv
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 12: GENERATING submission.csv")
print("=" * 70)

# The final_model pipeline already contains the fitted preprocessor, so we
# simply call predict() on the raw (engineered) test features.
test_predictions = final_model.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": test_predictions,
})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Predictions saved to: {submission_path}")
print(submission.head())


# =============================================================================
# SECTION 13: SAVE THE TRAINED MODEL WITH joblib
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 13: SAVING THE TRAINED MODEL")
print("=" * 70)

model_path = os.path.join(OUTPUT_DIR, "best_titanic_model.joblib")
joblib.dump(final_model, model_path)
print(f"Trained pipeline saved to: {model_path}")
print("(Reload it later with: joblib.load('outputs/best_titanic_model.joblib'))")


# =============================================================================
# SECTION 14: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Best model selected     : {best_model_name}")
print(f"Best hyperparameters    : {grid_search.best_params_}")
print(f"Final validation accuracy: {final_accuracy:.4f}")
print(f"Final 5-fold CV accuracy : {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std():.4f})")
print(f"Final validation ROC-AUC : {final_roc_auc:.4f}")
print("All outputs written to the ./outputs/ directory.")
print("=" * 70)
