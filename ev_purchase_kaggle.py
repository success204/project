import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(df.head())
print(df.shape)
print(df.dtypes)
print(df["Will_Buy_EV"].value_counts())

print(f"\nDataset Information:")
df.info()

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nDataset Summary:")
print(df.describe())


# Target
y = df["Will_Buy_EV"].map({"Yes": 1, "No": 0})

# Features
X = df.drop(columns=["Will_Buy_EV", "id"])
X_test = test.drop(columns=["id"])

# Split the training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text columns to numbers
cat_cols = X.select_dtypes(include="object").columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_valid[cat_cols] = encoder.transform(X_valid[cat_cols])




# Create model
model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Validation prediction
pred = model.predict_proba(X_valid)[:, 1]

print("ROC-AUC:", roc_auc_score(y_valid, pred))
print("Accuracy:", accuracy_score(y_valid, pred >= 0.5))

#Encode full training and test data
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Train on all data

model.fit(X, y)

# Predict test data
test_pred = model.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": test_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission saved!")