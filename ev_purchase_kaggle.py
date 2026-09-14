import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier
)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\nFirst 5 rows:")
print(train.head())

print("\nData types:")
print(train.dtypes)

print("\nTarget distribution:")
print(train["Will_Buy_EV"].value_counts())

print("\nTarget percentage:")
print(train["Will_Buy_EV"].value_counts(normalize=True))

print("\nMissing values:")
print(train.isnull().sum())

print("\nDataset information:")
train.info()

print("\nNumerical summary:")
print(train.describe())


# ============================================================
# 3. TARGET
# ============================================================

y = train["Will_Buy_EV"].map({
    "Yes": 1,
    "No": 0
})

if y.isnull().any():
    raise ValueError(
        "Target contains values other than 'Yes' and 'No'."
    )


# ============================================================
# 4. FEATURES
# ============================================================

X = train.drop(
    columns=["Will_Buy_EV", "id"],
    errors="ignore"
).copy()

X_test = test.drop(
    columns=["id"],
    errors="ignore"
).copy()


# Make sure train and test contain the same features
missing_in_test = set(X.columns) - set(X_test.columns)

extra_in_test = set(X_test.columns) - set(X.columns)

if missing_in_test:
    raise ValueError(
        f"Columns missing from test data: {missing_in_test}"
    )

if extra_in_test:
    print(
        "\nDropping extra test columns:",
        extra_in_test
    )

    X_test = X_test.drop(
        columns=list(extra_in_test)
    )


# Same column order
X_test = X_test[X.columns]


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

numeric_cols = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_cols
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_cols
    )
])


# ============================================================
# 7. MODELS
# ============================================================

models = {

    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 8. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

test_predictions = {}

oof_predictions_all = {}


for model_name, classifier in models.items():

    print("\n")
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    oof_predictions = np.zeros(len(X))
    
    test_predictions_model = np.zeros(len(X_test))
    
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(f"\nTraining Fold {fold}/5...")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]


    # Create complete pipeline
        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                classifier
            )
        ])

    


        # Train
        pipeline.fit(
            X_train,
            y_train
        )


        # Validation predictions
        valid_pred = pipeline.predict_proba(
            X_valid
        )[:, 1]


        # Test predictions
        fold_test_pred = pipeline.predict_proba(
            X_test
        )[:, 1]


        # Store out-of-fold predictions
        oof_predictions[valid_idx] = valid_pred


        # Average test predictions
        test_predictions_model += (
            fold_test_pred / skf.n_splits
        )


        # Fold ROC-AUC
        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        fold_scores.append(fold_auc)

        print(
            f"Fold {fold} ROC-AUC: "
            f"{fold_auc:.6f}"
        )


    # ========================================================
    # OVERALL OOF PERFORMANCE
    # ========================================================

    overall_auc = roc_auc_score(
        y,
        oof_predictions
    )

    accuracy = accuracy_score(
        y,
        (oof_predictions >= 0.5).astype(int)
    )


    results[model_name] = {
        "auc": overall_auc,
        "accuracy": accuracy,
        "fold_scores": fold_scores
    }


    test_predictions[model_name] = (
        test_predictions_model
    )

    oof_predictions_all[model_name] = (
        oof_predictions
    )


    print("\nOverall ROC-AUC:")
    print(f"{overall_auc:.6f}")

    print("\nAccuracy @ 0.50:")
    print(f"{accuracy:.6f}")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for model_name, result in results.items():

    print(
        f"{model_name:25s} "
        f"ROC-AUC = {result['auc']:.6f} | "
        f"Accuracy = {result['accuracy']:.6f}"
    )


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["auc"]
)

print("\nBest model:")
print(best_model_name)

best_oof = oof_predictions_all[
    best_model_name
]

best_test_pred = test_predictions[
    best_model_name
]


# ============================================================
# 11. OPTIMIZE ACCURACY THRESHOLD
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)

best_threshold = 0.50
best_accuracy = 0

for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    predictions = (
        best_oof >= threshold
    ).astype(int)

    acc = accuracy_score(
        y,
        predictions
    )

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold


print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best OOF accuracy: "
    f"{best_accuracy:.6f}"
)


# ============================================================
# 12. FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

final_classifier = models[
    best_model_name
]

final_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        final_classifier
    )
])


# Train on ALL available training data
final_pipeline.fit(
    X,
    y
)


# ============================================================
# 13. FINAL TEST PREDICTIONS
# ============================================================

final_test_pred = final_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": final_test_pred
})


submission.to_csv(
    "submission.csv",
    index=False
)


# ============================================================
# 15. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    "Original baseline ROC-AUC : 0.940597"
)

print(
    f"Best CV ROC-AUC           : "
    f"{results[best_model_name]['auc']:.6f}"
)

print(
    f"Best model                : "
    f"{best_model_name}"
)

print(
    f"Best accuracy threshold   : "
    f"{best_threshold:.2f}"
)

print(
    f"Best OOF accuracy         : "
    f"{best_accuracy:.6f}"
)

print(
    "\nSubmission saved as:"
)

print(
    "submission.csv"
)


# ============================================================
# 16. PREVIEW SUBMISSION
# ============================================================

print("\nSubmission preview:")

print(
    submission.head(10)
)

# ============================================================
# TARGET RATE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("TARGET RATE ANALYSIS")
print("=" * 70)

analysis_df = train.copy()

analysis_df["target"] = (
    analysis_df["Will_Buy_EV"] == "Yes"
).astype(int)


# ------------------------------------------------------------
# CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_analysis_cols = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

for col in categorical_analysis_cols:

    print("\n" + "-" * 70)
    print(f"{col}")
    print("-" * 70)

    result = (
        analysis_df
        .groupby(col)["target"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    result["purchase_rate_%"] = result["mean"] * 100

    print(result)


# ------------------------------------------------------------
# NUMERICAL FEATURES
# ------------------------------------------------------------

numeric_analysis_cols = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level"
]

for col in numeric_analysis_cols:

    print("\n" + "-" * 70)
    print(f"{col}")
    print("-" * 70)

    analysis_df["bin"] = pd.qcut(
        analysis_df[col],
        q=10,
        duplicates="drop"
    )

    result = (
        analysis_df
        .groupby("bin", observed=True)["target"]
        .agg(["mean", "count"])
    )

    result["purchase_rate_%"] = result["mean"] * 100

    print(result)


# Cleanup
analysis_df.drop(
    columns=["target", "bin"],
    inplace=True,
    errors="ignore"
)


# ============================================================
# STAGE 2 — FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 — FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# 1. CREATE COPIES
# ------------------------------------------------------------

train_fe = train.copy()
test_fe = test.copy()


# ------------------------------------------------------------
# 2. BASIC NUMERICAL FEATURES
# ------------------------------------------------------------

# Total charging infrastructure available
train_fe["Total_Charging_Stations"] = (
    train_fe["Charging_Stations_Near_Home"]
    + train_fe["Charging_Stations_Near_Work"]
)

test_fe["Total_Charging_Stations"] = (
    test_fe["Charging_Stations_Near_Home"]
    + test_fe["Charging_Stations_Near_Work"]
)


# Difference between work and home charging availability
train_fe["Charging_Station_Gap"] = (
    train_fe["Charging_Stations_Near_Work"]
    - train_fe["Charging_Stations_Near_Home"]
)

test_fe["Charging_Station_Gap"] = (
    test_fe["Charging_Stations_Near_Work"]
    - test_fe["Charging_Stations_Near_Home"]
)


# ------------------------------------------------------------
# 3. INCOME-BASED FEATURES
# ------------------------------------------------------------

# Income relative to age
train_fe["Income_per_Age"] = (
    train_fe["Annual_Income_USD"] / train_fe["Age"]
)

test_fe["Income_per_Age"] = (
    test_fe["Annual_Income_USD"] / test_fe["Age"]
)


# Income per car owned
train_fe["Income_per_Car"] = (
    train_fe["Annual_Income_USD"]
    / train_fe["Number_of_Cars_Owned"].clip(lower=1)
)

test_fe["Income_per_Car"] = (
    test_fe["Annual_Income_USD"]
    / test_fe["Number_of_Cars_Owned"].clip(lower=1)
)


# ------------------------------------------------------------
# 4. COMMUTE FEATURES
# ------------------------------------------------------------

# Income relative to commute distance
train_fe["Income_per_Commute"] = (
    train_fe["Annual_Income_USD"]
    / train_fe["Daily_Commute_km"].clip(lower=1)
)

test_fe["Income_per_Commute"] = (
    test_fe["Annual_Income_USD"]
    / test_fe["Daily_Commute_km"].clip(lower=1)
)


# ------------------------------------------------------------
# 5. INTERACTION FEATURES
# ------------------------------------------------------------

# Environmental concern × income
train_fe["Environmental_Income"] = (
    train_fe["Environmental_Concern_Level"]
    * train_fe["Annual_Income_USD"]
)

test_fe["Environmental_Income"] = (
    test_fe["Environmental_Concern_Level"]
    * test_fe["Annual_Income_USD"]
)


# Environmental concern × charging access
train_fe["Environmental_Charging"] = (
    train_fe["Environmental_Concern_Level"]
    * train_fe["Total_Charging_Stations"]
)

test_fe["Environmental_Charging"] = (
    test_fe["Environmental_Concern_Level"]
    * test_fe["Total_Charging_Stations"]
)


# Commute × charging access
train_fe["Commute_Charging"] = (
    train_fe["Daily_Commute_km"]
    * train_fe["Total_Charging_Stations"]
)

test_fe["Commute_Charging"] = (
    test_fe["Daily_Commute_km"]
    * test_fe["Total_Charging_Stations"]
)


# Income × number of cars
train_fe["Income_Cars"] = (
    train_fe["Annual_Income_USD"]
    * train_fe["Number_of_Cars_Owned"]
)

test_fe["Income_Cars"] = (
    test_fe["Annual_Income_USD"]
    * test_fe["Number_of_Cars_Owned"]
)


# ------------------------------------------------------------
# 6. PRINT NEW FEATURES
# ------------------------------------------------------------

new_features = [
    "Total_Charging_Stations",
    "Charging_Station_Gap",
    "Income_per_Age",
    "Income_per_Car",
    "Income_per_Commute",
    "Environmental_Income",
    "Environmental_Charging",
    "Commute_Charging",
    "Income_Cars"
]

print("\nNew engineered features:")
for feature in new_features:
    print(" -", feature)


print("\nOriginal train shape:", train.shape)
print("Feature-engineered train shape:", train_fe.shape)


# ------------------------------------------------------------
# 7. PREPARE TARGET
# ------------------------------------------------------------

y_fe = train_fe["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})

X_fe = train_fe.drop(columns=["Will_Buy_EV"])
X_test_fe = test_fe.copy()


# ------------------------------------------------------------
# 8. IDENTIFY COLUMNS
# ------------------------------------------------------------

numeric_features_fe = X_fe.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features_fe = X_fe.select_dtypes(
    include=["object", "str"]
).columns.tolist()


print("\nNumerical features:", len(numeric_features_fe))
print("Categorical features:", len(categorical_features_fe))


# ------------------------------------------------------------
# 9. PREPROCESSING
# ------------------------------------------------------------

preprocessor_fe = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features_fe
        ),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_features_fe
        )
    ]
)


# ------------------------------------------------------------
# 10. FEATURE-ENGINEERED HISTGRADIENTBOOSTING
# ------------------------------------------------------------

hgb_fe = Pipeline(
    steps=[
        ("preprocessor", preprocessor_fe),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 11. 5-FOLD CROSS VALIDATION
# ------------------------------------------------------------

skf_fe = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_predictions_fe = np.zeros(len(X_fe))

print("\n" + "=" * 70)
print("FEATURE-ENGINEERED HISTGRADIENTBOOSTING")
print("=" * 70)

for fold, (train_idx, valid_idx) in enumerate(
    skf_fe.split(X_fe, y_fe), 1
):

    print(f"\nTraining Fold {fold}/5...")

    X_train_fold = X_fe.iloc[train_idx]
    X_valid_fold = X_fe.iloc[valid_idx]

    y_train_fold = y_fe.iloc[train_idx]
    y_valid_fold = y_fe.iloc[valid_idx]

    hgb_fe.fit(
        X_train_fold,
        y_train_fold
    )

    valid_pred = hgb_fe.predict_proba(
        X_valid_fold
    )[:, 1]

    oof_predictions_fe[valid_idx] = valid_pred

    fold_auc = roc_auc_score(
        y_valid_fold,
        valid_pred
    )

    print(
        f"Fold {fold} ROC-AUC: {fold_auc:.6f}"
    )


# ------------------------------------------------------------
# 12. OVERALL SCORE
# ------------------------------------------------------------

fe_auc = roc_auc_score(
    y_fe,
    oof_predictions_fe
)

fe_accuracy = accuracy_score(
    y_fe,
    (oof_predictions_fe >= 0.50).astype(int)
)


print("\n" + "=" * 70)
print("STAGE 2 RESULTS")
print("=" * 70)

print(
    f"Original best ROC-AUC : {0.941408:.6f}"
)

print(
    f"Feature-engineered ROC-AUC : {fe_auc:.6f}"
)

print(
    f"Feature-engineered Accuracy : {fe_accuracy:.6f}"
)


# ------------------------------------------------------------
# 13. IMPROVEMENT
# ------------------------------------------------------------

improvement = fe_auc - 0.941408

print(
    f"\nROC-AUC improvement: {improvement:+.6f}"
)

if improvement > 0:
    print("SUCCESS: Feature engineering improved the model!")
else:
    print("Feature engineering did not improve the benchmark.")