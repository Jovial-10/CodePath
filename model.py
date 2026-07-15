import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib


# Clean data here:
df = pd.read_csv('ai_student_impact_dataset.csv')

# Strip whitespace from text columns
text_cols = df.select_dtypes(include='object').columns
for c in text_cols:
    df[c] = df[c].astype(str).str.strip()

# Drop exact duplicate rows and duplicate Student_IDs, if any
df = df.drop_duplicates()
df = df.drop_duplicates(subset='Student_ID', keep='first')

# Binary target: did GPA improve?
df['Improved'] = (df['Post_Semester_GPA'] > df['Pre_Semester_GPA']).astype(int)


# part2
drop_cols = ['Student_ID', 'Pre_Semester_GPA', 'Post_Semester_GPA', 'Improved']
X = df.drop(columns=drop_cols)
y = df['Improved']

# Convert text columns into 0/1 columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Model 2: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:   ", recall_score(y_true, y_pred))
    print("F1:       ", f1_score(y_true, y_pred))


print("=" * 60)
print("TRAIN/TEST EVALUATION (honest, held-out performance)")
print("=" * 60)

evaluate("Logistic Regression", y_test, log_preds)
evaluate("Random Forest", y_test, rf_preds)

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, log_preds))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_preds))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))


# ==============================================================
# PART 3: FEATURE IMPORTANCE / WEIGHTS (exploratory, trained on full data)
# NOTE: These functions train on the FULL dataset (no held-out test set),
# so their accuracy numbers are NOT directly comparable to Part 2's
# held-out performance — training accuracy is naturally optimistic since
# the model is scored on data it already learned from. Use these only to
# see which features matter most, not as your headline model performance.
# ==============================================================


def train_logistic_regression_model(dataset, feature_columns, target_column):
    print(f"Rows: {len(dataset)}")
    print(f"Columns used: {feature_columns}")

    features = dataset[feature_columns].fillna(dataset[feature_columns].mean())
    target = dataset[target_column]

    model = LogisticRegression(max_iter=1000, class_weight='balanced').fit(features, target)

    feature_weights = pd.Series(model.coef_[0], index=feature_columns).sort_values(ascending=False)
    print("\nFeature weights (log-odds, full-data fit):")
    print(feature_weights)

    return model


def train_random_forest_model(dataset, target_column, excluded_columns=None):
    excluded_columns = (excluded_columns or []) + [target_column]

    # Keep all columns except excluded ones (numeric AND categorical)
    feature_columns = [col for col in dataset.columns if col not in excluded_columns]
    raw_features = dataset[feature_columns]

    # One-hot encode categorical columns, keep numeric columns as-is
    features = pd.get_dummies(raw_features, drop_first=True)
    features = features.fillna(features.mean(numeric_only=True))

    target = dataset[target_column]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    ).fit(features, target)

    training_score = model.score(features, target)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=features.columns
    ).sort_values(ascending=False)

    print(f"\nPredicting: {target_column}")
    print(f"Using {len(features.columns)} features (after encoding)")
    print(f"Training Accuracy (full-data fit): {training_score:.2f}")

    print("\nMost important prediction signals:")
    print(feature_importance.head(10))

    return model


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE EXPLORATION")
print("=" * 60)

improvement_features = [
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Tool_Diversity",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams"
]

log_reg_exploratory = train_logistic_regression_model(
    df, improvement_features, "Improved"
)

rf_exploratory = train_random_forest_model(
    df,
    target_column="Improved",
    excluded_columns=["Student_ID", "Pre_Semester_GPA", "Post_Semester_GPA"]
)

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
