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


# ============================================================
# MODEL DEVELOPMENT AND CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL DEVELOPMENT AND CROSS-VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. PREPARE FEATURE-ENGINEERED DATA
# ------------------------------------------------------------

# Remove target and ID from the feature set

X_model = X_fe.drop(
    columns=["Will_Buy_EV", "id"],
    errors="ignore"
).copy()

X_test_model = X_test_fe.drop(
    columns=["id"],
    errors="ignore"
).copy()


# Make sure train and test contain the same features

missing_in_test = set(X_model.columns) - set(X_test_model.columns)

extra_in_test = set(X_test_model.columns) - set(X_model.columns)


if missing_in_test:

    raise ValueError(
        f"Columns missing from test data: {missing_in_test}"
    )


if extra_in_test:

    print(
        "\nDropping extra test columns:",
        extra_in_test
    )

    X_test_model = X_test_model.drop(
        columns=list(extra_in_test)
    )


# Match test column order to training data

X_test_model = X_test_model[X_model.columns]


# Replace infinite values created by ratio features

X_model = X_model.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test_model = X_test_model.replace(
    [np.inf, -np.inf],
    np.nan
)


print("\nTraining shape:")
print(X_model.shape)

print("\nTest shape:")
print(X_test_model.shape)


# ------------------------------------------------------------
# 2. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numeric_features_model = X_model.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32"
    ]
).columns.tolist()


categorical_features_model = X_model.select_dtypes(
    include=[
        "object",
        "string",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features_model)

print(
    "\nNumber of numerical features:",
    len(numeric_features_model)
)


print("\nCategorical features:")
print(categorical_features_model)

print(
    "\nNumber of categorical features:",
    len(categorical_features_model)
)


# ------------------------------------------------------------
# 3. ROBUST PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline_model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_pipeline_model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])


model_preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline_model,
        numeric_features_model
    ),
    (
        "categorical",
        categorical_pipeline_model,
        categorical_features_model
    )
])


# ------------------------------------------------------------
# 4. CANDIDATE MODELS
# ------------------------------------------------------------

candidate_models = {

    "HistGradientBoosting":
    HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    ),

    "ExtraTrees":
    ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest":
    RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# 5. STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

model_skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


model_results = {}

model_oof_predictions = {}


# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE EACH MODEL
# ------------------------------------------------------------

for model_name, classifier in candidate_models.items():

    print("\n")
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)


    # OOF predictions for this model

    oof_predictions = np.zeros(
        len(X_model)
    )


    # Store each fold's ROC-AUC

    fold_scores = []


    # --------------------------------------------------------
    # 7. TRAIN EACH FOLD
    # --------------------------------------------------------

    for fold, (train_idx, valid_idx) in enumerate(
        model_skf.split(
            X_model,
            y_fe
        ),
        start=1
    ):

        print(
            f"\nTraining Fold {fold}/5..."
        )


        # Split training and validation data

        X_train_fold = X_model.iloc[
            train_idx
        ]

        X_valid_fold = X_model.iloc[
            valid_idx
        ]


        y_train_fold = y_fe.iloc[
            train_idx
        ]

        y_valid_fold = y_fe.iloc[
            valid_idx
        ]


        # ----------------------------------------------------
        # CREATE PIPELINE
        # ----------------------------------------------------

        model_pipeline = Pipeline([
            (
                "preprocessor",
                model_preprocessor
            ),
            (
                "model",
                classifier
            )
        ])


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model_pipeline.fit(
            X_train_fold,
            y_train_fold
        )


        # ----------------------------------------------------
        # VALIDATION PROBABILITIES
        # ----------------------------------------------------

        valid_pred = model_pipeline.predict_proba(
            X_valid_fold
        )[:, 1]


        # ----------------------------------------------------
        # STORE OOF PREDICTIONS
        # ----------------------------------------------------

        oof_predictions[
            valid_idx
        ] = valid_pred


        # ----------------------------------------------------
        # CALCULATE FOLD ROC-AUC
        # ----------------------------------------------------

        fold_auc = roc_auc_score(
            y_valid_fold,
            valid_pred
        )


        fold_scores.append(
            fold_auc
        )


        print(
            f"Fold {fold} ROC-AUC: "
            f"{fold_auc:.6f}"
        )


    # --------------------------------------------------------
    # 8. OVERALL OOF PERFORMANCE
    # --------------------------------------------------------

    overall_auc = roc_auc_score(
        y_fe,
        oof_predictions
    )


    accuracy_050 = accuracy_score(
        y_fe,
        (
            oof_predictions >= 0.50
        ).astype(int)
    )


    mean_fold_auc = np.mean(
        fold_scores
    )


    std_fold_auc = np.std(
        fold_scores
    )


    # --------------------------------------------------------
    # 9. STORE MODEL RESULTS
    # --------------------------------------------------------

    model_results[model_name] = {

        "auc": overall_auc,

        "accuracy": accuracy_050,

        "mean_fold_auc": mean_fold_auc,

        "std_fold_auc": std_fold_auc,

        "fold_scores": fold_scores
    }


    model_oof_predictions[
        model_name
    ] = oof_predictions


    # --------------------------------------------------------
    # 10. PRINT MODEL PERFORMANCE
    # --------------------------------------------------------

    print("\nOverall OOF ROC-AUC:")

    print(
        f"{overall_auc:.6f}"
    )


    print("\nAccuracy @ 0.50:")

    print(
        f"{accuracy_050:.6f}"
    )


    print("\nMean Fold ROC-AUC:")

    print(
        f"{mean_fold_auc:.6f}"
    )


    print("\nFold ROC-AUC Std:")

    print(
        f"{std_fold_auc:.6f}"
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


for model_name, result in model_results.items():

    print(
        f"{model_name:25s}"
        f"ROC-AUC = {result['auc']:.6f} | "
        f"Accuracy = {result['accuracy']:.6f} | "
        f"Mean Fold AUC = {result['mean_fold_auc']:.6f} | "
        f"Std = {result['std_fold_auc']:.6f}"
    )


# ============================================================
# 12. SELECT CURRENT BEST MODEL
# ============================================================

best_model_name = max(
    model_results,
    key=lambda name:
    model_results[name]["auc"]
)


best_model_auc = model_results[
    best_model_name
]["auc"]


best_model_oof = model_oof_predictions[
    best_model_name
]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    f"Best model: {best_model_name}"
)


print(
    f"Best OOF ROC-AUC: "
    f"{best_model_auc:.6f}"
)


# ============================================================
# 13. COMPARE AGAINST STAGE 2
# ============================================================

previous_auc = fe_auc


improvement = (
    best_model_auc - previous_auc
)


print("\n")
print("=" * 70)
print("IMPROVEMENT")
print("=" * 70)


print(
    f"Previous ROC-AUC: "
    f"{previous_auc:.6f}"
)


print(
    f"Current ROC-AUC: "
    f"{best_model_auc:.6f}"
)


print(
    f"Improvement: "
    f"{improvement:+.6f}"
)


if improvement > 0:

    print(
        "\nSUCCESS: The current models improved "
        "the previous benchmark."
    )

elif improvement == 0:

    print(
        "\nThe current models matched "
        "the previous benchmark."
    )

else:

    print(
        "\nThe current models did not improve "
        "the previous benchmark."
    )


# ============================================================
# 14. MODEL DEVELOPMENT COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL DEVELOPMENT COMPLETE")
print("=" * 70)


print(
    f"Best model: {best_model_name}"
)


print(
    f"Best ROC-AUC: "
    f"{best_model_auc:.6f}"
)


print(
    "\nNo final submission was created yet."
)


print(
    "The next step will focus on further "
    "model optimization."
)

# ============================================================
# STAGE 4 — FINAL MODEL TRAINING & KAGGLE SUBMISSION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 4 — FINAL MODEL TRAINING & KAGGLE SUBMISSION")
print("=" * 70)


# ------------------------------------------------------------
# 1. PREPARE FINAL TRAINING DATA
# ------------------------------------------------------------

print("\nPreparing final training data...")

# Remove target and ID from the feature set
X_final = X_fe.drop(
    columns=["Will_Buy_EV", "id"],
    errors="ignore"
).copy()

X_test_final = X_test_fe.drop(
    columns=["Will_Buy_EV", "id"],
    errors="ignore"
).copy()


# ------------------------------------------------------------
# 2. TARGET ENCODING
# ------------------------------------------------------------

print("\nEncoding target...")

# Use the original target column directly.
# This avoids relying on the modified `y` variable from Stage 3.

y_final = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})

# Check whether encoding produced missing values
if y_final.isnull().any():
    raise ValueError(
        "Target encoding failed. Unexpected values found in "
        "train['Will_Buy_EV']."
    )

y_final = y_final.astype(int)


print("\nFinal training shape:")
print(X_final.shape)

print("\nFinal test shape:")
print(X_test_final.shape)

print("\nTarget distribution:")
print(y_final.value_counts())

print("\nTarget rate:")
print(y_final.mean())


# ------------------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numeric_features = X_final.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_final.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)

print("\nNumber of numerical features:")
print(len(numeric_features))

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:")
print(len(categorical_features))


# ------------------------------------------------------------
# 4. HANDLE NON-FINITE NUMERICAL VALUES
# ------------------------------------------------------------

print("\nChecking numerical features for non-finite values...")

# Replace +inf and -inf with NaN.
# SimpleImputer below will then replace NaN values with
# the median calculated from the training data.

X_final[numeric_features] = X_final[numeric_features].replace(
    [np.inf, -np.inf],
    np.nan
)

X_test_final[numeric_features] = X_test_final[numeric_features].replace(
    [np.inf, -np.inf],
    np.nan
)

print("Non-finite numerical values handled.")


# ------------------------------------------------------------
# 5. BUILD FINAL PREPROCESSING PIPELINE
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
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
    ]
)


preprocessor_final = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 6. DEFINE FINAL HISTGRADIENTBOOSTING MODEL
# ------------------------------------------------------------

final_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)


# ------------------------------------------------------------
# 7. CREATE FINAL PIPELINE
# ------------------------------------------------------------

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor_final
        ),
        (
            "model",
            final_model
        )
    ]
)


# ------------------------------------------------------------
# 8. TRAIN FINAL MODEL ON ALL TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING FINAL HISTGRADIENTBOOSTING MODEL")
print("=" * 70)

print("\nTraining on 100% of the available training data...")

final_pipeline.fit(
    X_final,
    y_final
)

print("\nFinal model training completed successfully.")


# ------------------------------------------------------------
# 9. GENERATE TEST PROBABILITIES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

test_probabilities = final_pipeline.predict_proba(
    X_test_final
)[:, 1]


print("\nPrediction summary:")
print(
    pd.Series(test_probabilities).describe()
)


# ------------------------------------------------------------
# 10. CHECK PREDICTIONS
# ------------------------------------------------------------

print("\nFirst 10 predicted probabilities:")
print(test_probabilities[:10])

print("\nMinimum prediction:")
print(test_probabilities.min())

print("\nMaximum prediction:")
print(test_probabilities.max())

print("\nMean prediction:")
print(test_probabilities.mean())


# ------------------------------------------------------------
# 11. CREATE SUBMISSION DATAFRAME
# ------------------------------------------------------------

submission = pd.DataFrame({
    "id": X_test_fe["id"].values,
    "Will_Buy_EV": test_probabilities
})


# ------------------------------------------------------------
# 12. VALIDATE SUBMISSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUBMISSION VALIDATION")
print("=" * 70)

print("\nSubmission shape:")
print(submission.shape)

print("\nExpected test rows:")
print(len(X_test_fe))

print("\nSubmission columns:")
print(submission.columns.tolist())

print("\nMissing values:")
print(submission.isnull().sum())

print("\nDuplicate IDs:")
print(submission["id"].duplicated().sum())

print("\nID check:")
print(
    submission["id"].equals(
        X_test_fe["id"].reset_index(drop=True)
    )
)


# ------------------------------------------------------------
# 13. SAVE SUBMISSION FILE
# ------------------------------------------------------------

submission.to_csv(
    "submission.csv",
    index=False
)


print("\n" + "=" * 70)
print("SUBMISSION CREATED SUCCESSFULLY")
print("=" * 70)

print("\nFile:")
print("submission.csv")

print("\nSubmission shape:")
print(submission.shape)

print("\nSubmission preview:")
print(submission.head(10))


# ------------------------------------------------------------
# 14. FINAL FILE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUBMISSION CHECK")
print("=" * 70)

print("\nNumber of rows:", len(submission))
print("Number of columns:", len(submission.columns))
print("Missing values:", submission.isnull().sum().sum())
print("Duplicate IDs:", submission["id"].duplicated().sum())

print("\nPrediction range:")
print(
    f"Min = {submission['Will_Buy_EV'].min():.6f}"
)

print(
    f"Max = {submission['Will_Buy_EV'].max():.6f}"
)

print(
    f"Mean = {submission['Will_Buy_EV'].mean():.6f}"
)

print("\nFirst 10 submission rows:")
print(submission.head(10))

print("\n" + "=" * 70)
print("STAGE 4 COMPLETE")
print("=" * 70)

print("\nYour submission file is ready:")
print("submission.csv")

print("\nIMPORTANT:")
print("Submit the probability values in Will_Buy_EV.")
print("Do NOT convert them to Yes/No.")

# ============================================================
# STAGE 5 — MODEL OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 5 — MODEL OPTIMIZATION")
print("=" * 70)

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier
)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

target = "Will_Buy_EV"

y = train[target].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget rate:")
print(y.mean())


# ------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Charging features
    # --------------------------------------------------------

    df["Total_Charging_Stations"] = (
        df["Charging_Stations_Near_Home"]
        + df["Charging_Stations_Near_Work"]
    )

    df["Charging_Station_Gap"] = (
        df["Charging_Stations_Near_Work"]
        - df["Charging_Stations_Near_Home"]
    )

    # --------------------------------------------------------
    # Income relationships
    # --------------------------------------------------------

    df["Income_per_Age"] = (
        df["Annual_Income_USD"] /
        (df["Age"] + 1)
    )

    df["Income_per_Car"] = (
        df["Annual_Income_USD"] /
        (df["Number_of_Cars_Owned"] + 1)
    )

    df["Income_per_Commute"] = (
        df["Annual_Income_USD"] /
        (df["Daily_Commute_km"] + 1)
    )

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    df["Environmental_Income"] = (
        df["Environmental_Concern_Level"]
        * df["Annual_Income_USD"]
    )

    df["Environmental_Charging"] = (
        df["Environmental_Concern_Level"]
        * df["Total_Charging_Stations"]
    )

    df["Commute_Charging"] = (
        df["Daily_Commute_km"]
        * df["Total_Charging_Stations"]
    )

    df["Income_Cars"] = (
        df["Annual_Income_USD"]
        * (df["Number_of_Cars_Owned"] + 1)
    )

    return df


train_fe = create_features(train)
test_fe = create_features(test)

print("\nFeature-engineered train shape:")
print(train_fe.shape)

print("\nFeature-engineered test shape:")
print(test_fe.shape)


# ------------------------------------------------------------
# 4. REMOVE TARGET AND ID
# ------------------------------------------------------------

X = train_fe.drop(
    columns=[target, "id"],
    errors="ignore"
).copy()

X_test = test_fe.drop(
    columns=["id"],
    errors="ignore"
).copy()


# ------------------------------------------------------------
# 5. IDENTIFY FEATURES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nNumber of numerical features:",
      len(numerical_features))

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of categorical features:",
      len(categorical_features))


# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
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
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 7. PREPARE DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

X_processed = np.asarray(
    X_processed,
    dtype=np.float64
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float64
)

# Replace possible non-finite values
X_processed = np.nan_to_num(
    X_processed,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test_processed = np.nan_to_num(
    X_test_processed,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print("\nProcessed training shape:")
print(X_processed.shape)

print("\nProcessed test shape:")
print(X_test_processed.shape)


# ------------------------------------------------------------
# 8. CROSS-VALIDATION SETUP
# ------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 9. MODEL CONFIGURATIONS
# ------------------------------------------------------------

models = {

    "HGB_Optimized_1": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    ),

    "HGB_Optimized_2": HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=450,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    ),

    "HGB_Optimized_3": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    ),

    "HGB_Optimized_4": HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42
    )
}


# ------------------------------------------------------------
# 10. MODEL EVALUATION
# ------------------------------------------------------------

results = []

best_model_name = None
best_auc = -np.inf
best_oof_predictions = None


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    oof_predictions = np.zeros(len(X_processed))

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X_processed, y),
        start=1
    ):

        print(
            f"\nTraining Fold {fold}/5..."
        )

        X_train = X_processed[train_idx]
        X_valid = X_processed[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(
            X_train,
            y_train
        )

        valid_pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof_predictions[valid_idx] = valid_pred

        fold_auc = roc_auc_score(
            y_valid,
            valid_pred
        )

        fold_scores.append(fold_auc)

        print(
            f"Fold {fold} ROC-AUC: "
            f"{fold_auc:.6f}"
        )

    overall_auc = roc_auc_score(
        y,
        oof_predictions
    )

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)

    accuracy = accuracy_score(
        y,
        (oof_predictions >= 0.50).astype(int)
    )

    print("\nOverall OOF ROC-AUC:")
    print(f"{overall_auc:.6f}")

    print("\nMean Fold ROC-AUC:")
    print(f"{mean_auc:.6f}")

    print("\nFold ROC-AUC Std:")
    print(f"{std_auc:.6f}")

    print("\nAccuracy @ 0.50:")
    print(f"{accuracy:.6f}")

    results.append({
        "Model": model_name,
        "OOF_ROC_AUC": overall_auc,
        "Mean_Fold_AUC": mean_auc,
        "Std_Fold_AUC": std_auc,
        "Accuracy": accuracy
    })

    if overall_auc > best_auc:

        best_auc = overall_auc

        best_model_name = model_name

        best_oof_predictions = (
            oof_predictions.copy()
        )


# ------------------------------------------------------------
# 11. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="OOF_ROC_AUC",
    ascending=False
)

print("\n" + "=" * 70)
print("STAGE 5 MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 12. BEST MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best model:", best_model_name)

print(
    f"Best OOF ROC-AUC: "
    f"{best_auc:.6f}"
)


# ------------------------------------------------------------
# 13. COMPARE WITH STAGE 4
# ------------------------------------------------------------

stage4_auc = 0.941392

improvement = best_auc - stage4_auc

print("\n" + "=" * 70)
print("COMPARISON WITH STAGE 4")
print("=" * 70)

print(
    f"Stage 4 ROC-AUC : "
    f"{stage4_auc:.6f}"
)

print(
    f"Stage 5 ROC-AUC : "
    f"{best_auc:.6f}"
)

print(
    f"Improvement     : "
    f"{improvement:+.6f}"
)

if improvement > 0:

    print(
        "\nSUCCESS: Stage 5 improved "
        "the Stage 4 benchmark."
    )

else:

    print(
        "\nNo improvement over Stage 4."
    )

    print(
        "Keep the Stage 4 model/submission "
        "as the current benchmark."
    )


# ------------------------------------------------------------
# 14. THRESHOLD SEARCH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0.50
best_accuracy = 0.0

for threshold in thresholds:

    predictions = (
        best_oof_predictions >= threshold
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


# ------------------------------------------------------------
# 15. SAVE STAGE 5 RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    "stage5_model_results.csv",
    index=False
)

print("\nResults saved as:")
print("stage5_model_results.csv")


# ------------------------------------------------------------
# 16. FINAL DECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 5 COMPLETE")
print("=" * 70)

if improvement > 0:

    print(
        "A new model has improved the "
        "Stage 4 benchmark."
    )

    print(
        "Next step: train the winning "
        "Stage 5 model on 100% of the data "
        "and create a new submission."
    )

else:

    print(
        "No model improved Stage 4."
    )

    print(
        "The Stage 4 submission remains "
        "the current benchmark."
    )

print("\nBest model:", best_model_name)
print(f"Best ROC-AUC: {best_auc:.6f}")
print(f"Stage 4 ROC-AUC: {stage4_auc:.6f}")
print(f"Improvement: {improvement:+.6f}")

print("\n" + "=" * 70)

# ============================================================
# STAGE 6 — XGBOOST ALTERNATIVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 6 — XGBOOST ALTERNATIVE MODEL")
print("=" * 70)

# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "\nXGBoost is not installed.\n"
        "Install it with:\n"
        "pip install xgboost\n"
    )


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

RANDOM_STATE = 42
N_SPLITS = 5


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ------------------------------------------------------------
# 4. TARGET ENCODING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET ANALYSIS")
print("=" * 70)

print("\nOriginal target distribution:")
print(train[TARGET].value_counts())

print("\nOriginal target percentages:")
print(
    train[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Convert:
# No  -> 0
# Yes -> 1

y = train[TARGET].map({
    "No": 0,
    "Yes": 1
})

# Check for unexpected values
if y.isna().any():
    print("\nERROR: Unexpected target values:")
    print(train.loc[y.isna(), TARGET].value_counts())
    raise ValueError(
        "TARGET must contain only 'Yes' and 'No'."
    )

y = y.astype(np.int8)

print("\nEncoded target rate:")
print(f"{y.mean():.4f} ({y.mean() * 100:.2f}%)")


# ------------------------------------------------------------
# 5. REMOVE ID
# ------------------------------------------------------------

X = train.drop(
    columns=[TARGET, ID_COL]
).copy()

X_test = test.drop(
    columns=[ID_COL]
).copy()

print("\nInitial feature shape:")
print(X.shape)

print("\nInitial test feature shape:")
print(X_test.shape)


# ------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


def feature_engineering(df):

    df = df.copy()

    # --------------------------------------------------------
    # Charging infrastructure
    # --------------------------------------------------------

    df["Total_Charging_Stations"] = (
        df["Charging_Stations_Near_Home"]
        + df["Charging_Stations_Near_Work"]
    )

    df["Charging_Station_Gap"] = (
        df["Charging_Stations_Near_Home"]
        - df["Charging_Stations_Near_Work"]
    )

    # --------------------------------------------------------
    # Income relationships
    # --------------------------------------------------------

    df["Income_per_Age"] = (
        df["Annual_Income_USD"]
        / (df["Age"] + 1)
    )

    df["Income_per_Car"] = (
        df["Annual_Income_USD"]
        / (df["Number_of_Cars_Owned"] + 1)
    )

    df["Income_per_Commute"] = (
        df["Annual_Income_USD"]
        / (df["Daily_Commute_km"] + 1)
    )

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    df["Environmental_Income"] = (
        df["Environmental_Concern_Level"]
        * df["Annual_Income_USD"]
    )

    df["Environmental_Charging"] = (
        df["Environmental_Concern_Level"]
        * df["Total_Charging_Stations"]
    )

    df["Commute_Charging"] = (
        df["Daily_Commute_km"]
        * df["Total_Charging_Stations"]
    )

    df["Income_Cars"] = (
        df["Annual_Income_USD"]
        * (df["Number_of_Cars_Owned"] + 1)
    )

    return df


X = feature_engineering(X)
X_test = feature_engineering(X_test)

print("\nFeature-engineered train shape:")
print(X.shape)

print("\nFeature-engineered test shape:")
print(X_test.shape)


# ------------------------------------------------------------
# 7. IDENTIFY CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 8. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 70)

for col in categorical_features:

    # Combine train and test so both use identical mappings
    combined = pd.concat(
        [
            X[col],
            X_test[col]
        ],
        axis=0
    ).astype(str)

    categories = pd.Categorical(
        combined
    ).categories

    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    X[col] = (
        X[col]
        .astype(str)
        .map(mapping)
        .astype(np.int32)
    )

    X_test[col] = (
        X_test[col]
        .astype(str)
        .map(mapping)
        .astype(np.int32)
    )


print("\nCategorical encoding completed.")


# ------------------------------------------------------------
# 9. NUMERIC CONVERSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING NUMERIC FEATURES")
print("=" * 70)

for col in X.columns:

    if col not in categorical_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# 10. HANDLE INFINITE VALUES
# ------------------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ------------------------------------------------------------
# 11. HANDLE MISSING VALUES
# ------------------------------------------------------------

for col in X.columns:

    median_value = X[col].median()

    if pd.isna(median_value):
        median_value = 0

    X[col] = X[col].fillna(
        median_value
    )

    X_test[col] = X_test[col].fillna(
        median_value
    )


print("\nFinal training shape:")
print(X.shape)

print("\nFinal test shape:")
print(X_test.shape)

print("\nMissing values in training:")
print(X.isna().sum().sum())

print("Missing values in test:")
print(X_test.isna().sum().sum())


# ------------------------------------------------------------
# 12. 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_predictions = np.zeros(
    len(X),
    dtype=np.float64
)

test_predictions = np.zeros(
    len(X_test),
    dtype=np.float64
)

fold_scores = []


# ------------------------------------------------------------
# 13. XGBOOST TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING XGBOOST MODELS")
print("=" * 70)

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print(
        f"\nTraining Fold "
        f"{fold}/{N_SPLITS}..."
    )

    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]

    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]


    model = XGBClassifier(

        n_estimators=1000,

        learning_rate=0.035,

        max_depth=6,

        min_child_weight=3,

        subsample=0.85,

        colsample_bytree=0.85,

        gamma=0.0,

        reg_alpha=0.05,

        reg_lambda=1.5,

        objective="binary:logistic",

        eval_metric="auc",

        tree_method="hist",

        random_state=RANDOM_STATE + fold,

        n_jobs=-1,

        verbosity=0
    )


    model.fit(
        X_train_fold,
        y_train_fold,

        eval_set=[
            (
                X_valid_fold,
                y_valid_fold
            )
        ],

        verbose=False
    )


    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    valid_pred = model.predict_proba(
        X_valid_fold
    )[:, 1]


    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_pred = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Store OOF predictions
    # --------------------------------------------------------

    oof_predictions[valid_idx] = valid_pred


    # Average predictions across folds

    test_predictions += (
        test_pred / N_SPLITS
    )


    # --------------------------------------------------------
    # Fold ROC-AUC
    # --------------------------------------------------------

    fold_auc = roc_auc_score(
        y_valid_fold,
        valid_pred
    )

    fold_scores.append(
        fold_auc
    )


    print(
        f"Fold {fold} ROC-AUC: "
        f"{fold_auc:.6f}"
    )


# ------------------------------------------------------------
# 14. OVERALL OOF RESULTS
# ------------------------------------------------------------

overall_auc = roc_auc_score(
    y,
    oof_predictions
)

mean_fold_auc = np.mean(
    fold_scores
)

std_fold_auc = np.std(
    fold_scores
)

accuracy = accuracy_score(
    y,
    (
        oof_predictions >= 0.50
    ).astype(int)
)


# ------------------------------------------------------------
# 15. STAGE 6 RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 6 XGBOOST RESULTS")
print("=" * 70)

print(
    f"\nOverall OOF ROC-AUC: "
    f"{overall_auc:.6f}"
)

print(
    f"Mean Fold ROC-AUC: "
    f"{mean_fold_auc:.6f}"
)

print(
    f"Fold ROC-AUC Std: "
    f"{std_fold_auc:.6f}"
)

print(
    f"Accuracy @ 0.50: "
    f"{accuracy:.6f}"
)


# ------------------------------------------------------------
# 16. COMPARE WITH STAGE 4 BENCHMARK
# ------------------------------------------------------------

STAGE4_AUC = 0.941392

improvement = (
    overall_auc
    - STAGE4_AUC
)

print("\n" + "=" * 70)
print("COMPARISON WITH STAGE 4")
print("=" * 70)

print(
    f"\nStage 4 ROC-AUC : "
    f"{STAGE4_AUC:.6f}"
)

print(
    f"Stage 6 ROC-AUC : "
    f"{overall_auc:.6f}"
)

print(
    f"Difference      : "
    f"{improvement:+.6f}"
)


if overall_auc > STAGE4_AUC:

    print(
        "\nStage 6 produced a higher ROC-AUC "
        "than the Stage 4 benchmark."
    )

elif overall_auc < STAGE4_AUC:

    print(
        "\nStage 6 produced a lower ROC-AUC "
        "than the Stage 4 benchmark."
    )

else:

    print(
        "\nStage 6 matched the Stage 4 benchmark."
    )


# ------------------------------------------------------------
# 17. SAVE MODEL RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({

    "Model": [
        "XGBoost_Stage6"
    ],

    "OOF_ROC_AUC": [
        overall_auc
    ],

    "Mean_Fold_AUC": [
        mean_fold_auc
    ],

    "Std_Fold_AUC": [
        std_fold_auc
    ],

    "Accuracy": [
        accuracy
    ],

    "Stage4_AUC": [
        STAGE4_AUC
    ],

    "Difference_vs_Stage4": [
        improvement
    ]
})


results.to_csv(
    "stage6_model_results.csv",
    index=False
)


# ------------------------------------------------------------
# 18. SAVE OOF PREDICTIONS
# ------------------------------------------------------------

oof_output = pd.DataFrame({

    ID_COL:
        train[ID_COL],

    TARGET:
        y,

    "OOF_Prediction":
        oof_predictions
})


oof_output.to_csv(
    "stage6_oof_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 19. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

test_output = pd.DataFrame({

    ID_COL:
        test[ID_COL],

    "Prediction":
        test_predictions
})


test_output.to_csv(
    "stage6_test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 20. OPTIONAL SUBMISSION FILE
# ------------------------------------------------------------

submission = pd.DataFrame({

    ID_COL:
        test[ID_COL],

    TARGET:
        np.where(
            test_predictions >= 0.50,
            "Yes",
            "No"
        )
})


submission.to_csv(
    "submission_stage6_xgboost.csv",
    index=False
)


# ------------------------------------------------------------
# 21. PREDICTION DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    submission[TARGET]
    .value_counts()
)

print("\nPrediction percentages:")

print(
    submission[TARGET]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 22. COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 6 COMPLETE")
print("=" * 70)

print(
    f"\nStage 6 XGBoost OOF ROC-AUC: "
    f"{overall_auc:.6f}"
)

print(
    f"Stage 4 benchmark: "
    f"{STAGE4_AUC:.6f}"
)

print(
    f"Difference vs Stage 4: "
    f"{improvement:+.6f}"
)

print("\nFiles saved:")

print(
    " - stage6_model_results.csv"
)

print(
    " - stage6_oof_predictions.csv"
)

print(
    " - stage6_test_predictions.csv"
)

print(
    " - submission_stage6_xgboost.csv"
)


# ============================================================
# STAGE 7 — FINAL XGBOOST MODEL & KAGGLE SUBMISSION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 7 — FINAL XGBOOST MODEL & KAGGLE SUBMISSION")
print("=" * 70)


# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 2. FILE NAMES
# ============================================================

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET = "Will_Buy_EV"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_data.shape}")
print(f"Test shape : {test_data.shape}")


# ============================================================
# 4. CHECK TARGET
# ============================================================

if TARGET not in train_data.columns:
    raise ValueError(
        f"'{TARGET}' was not found in the training dataset."
    )

print("\nTarget distribution:")
print(train_data[TARGET].value_counts())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = train_data.drop(columns=[TARGET]).copy()
y = train_data[TARGET].copy()

X_test = test_data.copy()


# ============================================================
# 6. REMOVE UNNECESSARY INDEX COLUMNS
# ============================================================

columns_to_remove = []

for column in X.columns:

    if str(column).lower().startswith("unnamed:"):
        columns_to_remove.append(column)

if columns_to_remove:

    X = X.drop(columns=columns_to_remove)

    X_test = X_test.drop(
        columns=[
            column
            for column in columns_to_remove
            if column in X_test.columns
        ]
    )


# ============================================================
# 7. MAKE TRAIN AND TEST COLUMNS MATCH
# ============================================================

common_columns = [
    column
    for column in X.columns
    if column in X_test.columns
]

X = X[common_columns]
X_test = X_test[common_columns]

print("\nNumber of features:", len(common_columns))


# ============================================================
# 8. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

numeric_columns = X.select_dtypes(
    include=["int64", "int32", "float64", "float32", "bool"]
).columns.tolist()

categorical_columns = [
    column
    for column in X.columns
    if column not in numeric_columns
]

print("Numeric columns    :", len(numeric_columns))
print("Categorical columns:", len(categorical_columns))


# ============================================================
# 9. HANDLE NUMERIC MISSING VALUES
# ============================================================

for column in numeric_columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )

    median_value = X[column].median()

    X[column] = X[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


# ============================================================
# 10. HANDLE CATEGORICAL COLUMNS
# ============================================================

for column in categorical_columns:

    X[column] = X[column].fillna("Missing").astype(str)
    X_test[column] = X_test[column].fillna("Missing").astype(str)


# ============================================================
# 11. ENCODE CATEGORICAL FEATURES
# ============================================================

if len(categorical_columns) > 0:

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X[categorical_columns] = encoder.fit_transform(
        X[categorical_columns]
    )

    X_test[categorical_columns] = encoder.transform(
        X_test[categorical_columns]
    )


# ============================================================
# 12. CONVERT FEATURES TO FLOAT32
# ============================================================

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)


# ============================================================
# 13. ENCODE TARGET
# ============================================================

target_values = sorted(
    y.unique()
)

if len(target_values) != 2:
    raise ValueError(
        "Will_Buy_EV must contain exactly two classes."
    )

target_mapping = {
    target_values[0]: 0,
    target_values[1]: 1
}

y_encoded = y.map(target_mapping).astype(int)

print("\nTarget mapping:")
print(target_mapping)


# ============================================================
# 14. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION SET")
print("=" * 70)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training rows  :", len(X_train))
print("Validation rows:", len(X_valid))


# ============================================================
# 15. HANDLE CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\nClass 0:", negative_count)
print("Class 1:", positive_count)
print(
    "Scale positive weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# 16. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

print("XGBoost training completed.")


# ============================================================
# 17. VALIDATION PREDICTIONS
# ============================================================

validation_probabilities = model.predict_proba(
    X_valid
)[:, 1]

validation_predictions = (
    validation_probabilities >= 0.50
).astype(int)


# ============================================================
# 18. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_valid,
    validation_predictions
)

precision = precision_score(
    y_valid,
    validation_predictions,
    zero_division=0
)

recall = recall_score(
    y_valid,
    validation_predictions,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    validation_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_valid,
    validation_probabilities
)


print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        validation_predictions,
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_valid,
        validation_predictions
    )
)


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 22. RETRAIN ON FULL TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON FULL DATA")
print("=" * 70)

final_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

final_model.fit(
    X,
    y_encoded,
    verbose=False
)

print("Final model trained.")


# ============================================================
# 23. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEST PREDICTIONS")
print("=" * 70)

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

test_predictions_encoded = (
    test_probabilities >= 0.50
).astype(int)


# ============================================================
# 24. CONVERT BACK TO ORIGINAL TARGET VALUES
# ============================================================

reverse_mapping = {
    0: target_values[0],
    1: target_values[1]
}

test_predictions = pd.Series(
    test_predictions_encoded
).map(reverse_mapping)


# ============================================================
# 25. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    TARGET: test_predictions
})


# ============================================================
# 26. CHECK SUBMISSION
# ============================================================

print("\nSubmission shape:", submission.shape)

print("\nPrediction distribution:")
print(
    submission[TARGET].value_counts()
)

print(
    "\nMissing predictions:",
    submission[TARGET].isna().sum()
)

if len(submission) != len(test_data):
    raise ValueError(
        "Submission row count does not match test data."
    )

if submission[TARGET].isna().any():
    raise ValueError(
        "Submission contains missing predictions."
    )


# ============================================================
# 27. SAVE SUBMISSION
# ============================================================

submission_file = "submission_stage7.csv"

submission.to_csv(
    submission_file,
    index=False
)

print(
    f"\nSubmission saved successfully as: "
    f"{submission_file}"
)


# ============================================================
# 28. SHOW FIRST 10 PREDICTIONS
# ============================================================

print("\nFirst 10 predictions:")
print(
    submission.head(10)
)


# ============================================================
# 29. STAGE 7 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 7 COMPLETE")
print("=" * 70)

print(f"Validation Accuracy : {accuracy:.6f}")
print(f"Validation Precision: {precision:.6f}")
print(f"Validation Recall   : {recall:.6f}")
print(f"Validation F1       : {f1:.6f}")
print(f"Validation ROC-AUC  : {roc_auc:.6f}")

print(
    f"\nSubmission file: {submission_file}"
)

print("=" * 70)


# ============================================================
# STAGE 8 — XGBOOST OPTIMIZATION & FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 8 — XGBOOST OPTIMIZATION & FEATURE VALIDATION")
print("=" * 70)


# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 2. FILES AND TARGET
# ============================================================

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET = "Will_Buy_EV"

RANDOM_STATE = 42


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_data.shape}")
print(f"Test shape : {test_data.shape}")


# ============================================================
# 4. TARGET CHECK
# ============================================================

if TARGET not in train_data.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

print("\nTarget distribution:")
print(train_data[TARGET].value_counts())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X_original = train_data.drop(
    columns=[TARGET]
).copy()

y = train_data[TARGET].copy()

X_test_original = test_data.copy()


# ============================================================
# 6. REMOVE UNNAMED INDEX COLUMNS
# ============================================================

unnamed_columns = [
    column
    for column in X_original.columns
    if str(column).lower().startswith("unnamed:")
]

if unnamed_columns:

    print("\nRemoving unnamed columns:")
    print(unnamed_columns)

    X_original = X_original.drop(
        columns=unnamed_columns
    )

    X_test_original = X_test_original.drop(
        columns=[
            column
            for column in unnamed_columns
            if column in X_test_original.columns
        ]
    )


# ============================================================
# 7. CHECK FOR ID COLUMN
# ============================================================

id_columns = [
    column
    for column in X_original.columns
    if str(column).lower() == "id"
]

print("\nID columns detected:")
print(id_columns)


# ============================================================
# 8. TARGET ENCODING
# ============================================================

target_values = sorted(
    y.unique()
)

if len(target_values) != 2:
    raise ValueError(
        "The target must contain exactly two classes."
    )

target_mapping = {
    target_values[0]: 0,
    target_values[1]: 1
}

y_encoded = y.map(
    target_mapping
).astype(int)

print("\nTarget mapping:")
print(target_mapping)


# ============================================================
# 9. CREATE TWO FEATURE SETS
# ============================================================

# ------------------------------------------------------------
# Version A — all features
# ------------------------------------------------------------

X_with_id = X_original.copy()

X_test_with_id = X_test_original.copy()


# ------------------------------------------------------------
# Version B — remove ID
# ------------------------------------------------------------

X_without_id = X_original.drop(
    columns=id_columns,
    errors="ignore"
).copy()

X_test_without_id = X_test_original.drop(
    columns=id_columns,
    errors="ignore"
).copy()


print("\nFeature counts:")
print(
    "With ID   :",
    X_with_id.shape[1]
)

print(
    "Without ID:",
    X_without_id.shape[1]
)


# ============================================================
# 10. FUNCTION FOR FEATURE PREPARATION
# ============================================================

def prepare_features(X, X_test):

    X = X.copy()
    X_test = X_test.copy()

    # --------------------------------------------------------
    # Align train and test columns
    # --------------------------------------------------------

    common_columns = [
        column
        for column in X.columns
        if column in X_test.columns
    ]

    X = X[common_columns].copy()
    X_test = X_test[common_columns].copy()

    # --------------------------------------------------------
    # Identify column types
    # --------------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32",
            "bool"
        ]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    # --------------------------------------------------------
    # Numeric missing values
    # --------------------------------------------------------

    for column in numeric_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce"
        )

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

        X_test[column] = X_test[column].fillna(
            median_value
        )

    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    for column in categorical_columns:

        X[column] = (
            X[column]
            .fillna("Missing")
            .astype(str)
        )

        X_test[column] = (
            X_test[column]
            .fillna("Missing")
            .astype(str)
        )

    # --------------------------------------------------------
    # Encode categorical variables
    # --------------------------------------------------------

    if len(categorical_columns) > 0:

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X[categorical_columns] = (
            encoder.fit_transform(
                X[categorical_columns]
            )
        )

        X_test[categorical_columns] = (
            encoder.transform(
                X_test[categorical_columns]
            )
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X, X_test


# ============================================================
# 11. PREPARE BOTH FEATURE SETS
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FEATURES")
print("=" * 70)

X_with_id, X_test_with_id = prepare_features(
    X_with_id,
    X_test_with_id
)

X_without_id, X_test_without_id = prepare_features(
    X_without_id,
    X_test_without_id
)

print(
    "\nWith ID feature shape:",
    X_with_id.shape
)

print(
    "Without ID feature shape:",
    X_without_id.shape
)


# ============================================================
# 12. SAME VALIDATION INDICES FOR EVERY MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING CONSISTENT VALIDATION SPLIT")
print("=" * 70)

train_indices, valid_indices = train_test_split(
    np.arange(len(y_encoded)),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(
    f"Training rows  : {len(train_indices):,}"
)

print(
    f"Validation rows: {len(valid_indices):,}"
)


# ============================================================
# 13. CLASS IMBALANCE
# ============================================================

y_train = y_encoded.iloc[train_indices]

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count / positive_count
)

print("\nClass 0:", negative_count)
print("Class 1:", positive_count)

print(
    "Scale positive weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# 14. STAGE 7 BENCHMARK
# ============================================================

STAGE7_ACCURACY = 0.858360
STAGE7_PRECISION = 0.558529
STAGE7_RECALL = 0.901738
STAGE7_F1 = 0.689801
STAGE7_ROC_AUC = 0.940774

print("\n" + "=" * 70)
print("STAGE 7 BENCHMARK")
print("=" * 70)

print(
    f"Accuracy : {STAGE7_ACCURACY:.6f}"
)

print(
    f"Precision: {STAGE7_PRECISION:.6f}"
)

print(
    f"Recall   : {STAGE7_RECALL:.6f}"
)

print(
    f"F1 Score : {STAGE7_F1:.6f}"
)

print(
    f"ROC-AUC  : {STAGE7_ROC_AUC:.6f}"
)


# ============================================================
# 15. MODEL CONFIGURATIONS
# ============================================================

models_to_test = [

    {
        "name": "Baseline_No_ID",
        "features": "without_id",
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0
    },

    {
        "name": "Tuned_No_ID",
        "features": "without_id",
        "n_estimators": 650,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 3,
        "gamma": 0.05,
        "reg_alpha": 0.10,
        "reg_lambda": 1.50
    },

    {
        "name": "Regularized_No_ID",
        "features": "without_id",
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 5,
        "gamma": 0.10,
        "reg_alpha": 0.15,
        "reg_lambda": 2.00
    },

    {
        "name": "Tuned_With_ID",
        "features": "with_id",
        "n_estimators": 650,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 3,
        "gamma": 0.05,
        "reg_alpha": 0.10,
        "reg_lambda": 1.50
    }
]


# ============================================================
# 16. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}


for config in models_to_test:

    print("\n" + "=" * 70)

    print(
        "TESTING MODEL:",
        config["name"]
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    if config["features"] == "with_id":

        X_current = X_with_id

    else:

        X_current = X_without_id

    X_train_current = X_current.iloc[
        train_indices
    ]

    X_valid_current = X_current.iloc[
        valid_indices
    ]

    y_train_current = y_encoded.iloc[
        train_indices
    ]

    y_valid_current = y_encoded.iloc[
        valid_indices
    ]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    current_model = XGBClassifier(

        n_estimators=config["n_estimators"],

        max_depth=config["max_depth"],

        learning_rate=config["learning_rate"],

        subsample=config["subsample"],

        colsample_bytree=config[
            "colsample_bytree"
        ],

        min_child_weight=config[
            "min_child_weight"
        ],

        gamma=config["gamma"],

        reg_alpha=config["reg_alpha"],

        reg_lambda=config["reg_lambda"],

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        scale_pos_weight=scale_pos_weight
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    current_model.fit(
        X_train_current,
        y_train_current,
        eval_set=[
            (
                X_valid_current,
                y_valid_current
            )
        ],
        verbose=False
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = current_model.predict_proba(
        X_valid_current
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    current_accuracy = accuracy_score(
        y_valid_current,
        predictions
    )

    current_precision = precision_score(
        y_valid_current,
        predictions,
        zero_division=0
    )

    current_recall = recall_score(
        y_valid_current,
        predictions,
        zero_division=0
    )

    current_f1 = f1_score(
        y_valid_current,
        predictions,
        zero_division=0
    )

    current_auc = roc_auc_score(
        y_valid_current,
        probabilities
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Model": config["name"],

        "Features": config["features"],

        "Accuracy": current_accuracy,

        "Precision": current_precision,

        "Recall": current_recall,

        "F1": current_f1,

        "ROC_AUC": current_auc

    })

    trained_models[
        config["name"]
    ] = current_model

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nAccuracy : {current_accuracy:.6f}"
    )

    print(
        f"Precision: {current_precision:.6f}"
    )

    print(
        f"Recall   : {current_recall:.6f}"
    )

    print(
        f"F1 Score : {current_f1:.6f}"
    )

    print(
        f"ROC-AUC  : {current_auc:.6f}"
    )


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 18. COMPARE AGAINST STAGE 7
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON AGAINST STAGE 7")
print("=" * 70)

best_model_name = results_df.iloc[0]["Model"]

best_auc = results_df.iloc[0]["ROC_AUC"]

best_accuracy = results_df.iloc[0]["Accuracy"]

best_precision = results_df.iloc[0]["Precision"]

best_recall = results_df.iloc[0]["Recall"]

best_f1 = results_df.iloc[0]["F1"]


print(
    "\nBest Stage 8 model:",
    best_model_name
)

print(
    f"\nStage 7 ROC-AUC : "
    f"{STAGE7_ROC_AUC:.6f}"
)

print(
    f"Stage 8 ROC-AUC : "
    f"{best_auc:.6f}"
)

print(
    f"Difference      : "
    f"{best_auc - STAGE7_ROC_AUC:+.6f}"
)


print(
    f"\nStage 7 Accuracy: "
    f"{STAGE7_ACCURACY:.6f}"
)

print(
    f"Stage 8 Accuracy: "
    f"{best_accuracy:.6f}"
)


# ============================================================
# 19. BEST MODEL DETAILED EVALUATION
# ============================================================

best_model = trained_models[
    best_model_name
]

if (
    results_df.iloc[0]["Features"]
    == "with_id"
):

    best_X = X_with_id

else:

    best_X = X_without_id


best_X_valid = best_X.iloc[
    valid_indices
]

best_y_valid = y_encoded.iloc[
    valid_indices
]

best_probabilities = best_model.predict_proba(
    best_X_valid
)[:, 1]

best_predictions = (
    best_probabilities >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("BEST MODEL DETAILED EVALUATION")
print("=" * 70)

print("\nClassification Report:")

print(
    classification_report(
        best_y_valid,
        best_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        best_y_valid,
        best_predictions
    )
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame({

    "Feature": best_X.columns,

    "Importance": best_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 21. SAVE STAGE 8 RESULTS
# ============================================================

results_df.to_csv(
    "stage8_model_comparison.csv",
    index=False
)

importance_df.to_csv(
    "stage8_feature_importance.csv",
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 8 COMPLETE")
print("=" * 70)

print(
    "\nBest model:",
    best_model_name
)

print(
    f"Best ROC-AUC : {best_auc:.6f}"
)

print(
    f"Best Accuracy: {best_accuracy:.6f}"
)

print(
    f"Best Precision: {best_precision:.6f}"
)

print(
    f"Best Recall: {best_recall:.6f}"
)

print(
    f"Best F1: {best_f1:.6f}"
)

print(
    "\nFiles created:"
)

print(
    " - stage8_model_comparison.csv"
)

print(
    " - stage8_feature_importance.csv"
)

print("\nDo NOT create the final submission yet.")

print(
    "Use the Stage 8 results to determine the final model "
    "configuration for Stage 9."
)

print("=" * 70)


# ======================================================================
# STAGE 9 — FINAL XGBOOST MODEL TRAINING & PREDICTION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 9 — FINAL XGBOOST MODEL TRAINING & PREDICTION")
print("=" * 70)

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

# Automatically find train/test files
possible_paths = [
    ".",
    "data",
    "dataset",
    "input"
]

train_path = None
test_path = None

for path in possible_paths:
    train_candidate = os.path.join(path, "train.csv")
    test_candidate = os.path.join(path, "test.csv")

    if os.path.exists(train_candidate) and os.path.exists(test_candidate):
        train_path = train_candidate
        test_path = test_candidate
        break

if train_path is None:
    raise FileNotFoundError(
        "Could not find train.csv and test.csv. "
        "Place them in the project folder or update the file paths."
    )

print("Train file:", train_path)
print("Test file :", test_path)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape :", test.shape)


# ======================================================================
# 2. TARGET
# ======================================================================

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"

if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

y = train[TARGET].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in Will_Buy_EV.")

print("\nTarget distribution:")
print(train[TARGET].value_counts())

print("\nTarget mapping:")
print({"No": 0, "Yes": 1})


# ======================================================================
# 3. PREPARE FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("PREPARING FINAL FEATURES")
print("=" * 70)

# Remove target
X = train.drop(columns=[TARGET]).copy()
X_test = test.copy()

# Remove ID because Stage 8 showed that the selected model
# is Regularized_No_ID.
if ID_COLUMN in X.columns:
    X = X.drop(columns=[ID_COLUMN])

if ID_COLUMN in X_test.columns:
    X_test = X_test.drop(columns=[ID_COLUMN])

print("Training features:", X.shape)
print("Test features    :", X_test.shape)


# ======================================================================
# 4. COMBINE DATA FOR CONSISTENT ENCODING
# ======================================================================

print("\n" + "=" * 70)
print("ENCODING FEATURES")
print("=" * 70)

# Combine train and test temporarily so categorical columns receive
# exactly the same one-hot columns.
combined = pd.concat(
    [X, X_test],
    axis=0,
    ignore_index=True
)

categorical_columns = combined.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Categorical columns:")
print(categorical_columns)

# One-hot encode categorical variables
combined = pd.get_dummies(
    combined,
    columns=categorical_columns,
    dummy_na=True
)

# Convert boolean columns to integers
for column in combined.columns:
    if combined[column].dtype == bool:
        combined[column] = combined[column].astype(int)

# Replace infinite values
combined = combined.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing values using median calculated from the combined data.
# This guarantees that train and test have no missing values.
for column in combined.columns:
    if combined[column].isna().any():
        median_value = combined[column].median()

        if pd.isna(median_value):
            median_value = 0

        combined[column] = combined[column].fillna(median_value)

# Split back
X_final = combined.iloc[:len(X)].copy()
X_test_final = combined.iloc[len(X):].copy()

print("\nFinal training shape:", X_final.shape)
print("Final test shape    :", X_test_final.shape)

print("Missing values in training:",
      X_final.isna().sum().sum())

print("Missing values in test:",
      X_test_final.isna().sum().sum())


# ======================================================================
# 5. CLASS WEIGHT
# ======================================================================

negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\n" + "=" * 70)
print("CLASS BALANCING")
print("=" * 70)

print("Negative class:", negative_count)
print("Positive class:", positive_count)
print(
    "Scale positive weight:",
    round(scale_pos_weight, 4)
)


# ======================================================================
# 6. FINAL MODEL
# ======================================================================

print("\n" + "=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print("Selected model: Regularized_No_ID")

# Stage 8 selected the regularized no-ID configuration.
# These parameters are deliberately conservative to reduce overfitting.
final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print(final_model)


# ======================================================================
# 7. TRAIN ON ALL TRAINING DATA
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

print("Training rows:", len(X_final))
print("Training features:", X_final.shape[1])

final_model.fit(
    X_final,
    y
)

print("\nFinal model training complete.")


# ======================================================================
# 8. GENERATE TEST PROBABILITIES
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

test_probabilities = final_model.predict_proba(
    X_test_final
)[:, 1]

print("Prediction count:", len(test_probabilities))

print("\nPrediction statistics:")
print("Minimum :", round(test_probabilities.min(), 6))
print("Maximum :", round(test_probabilities.max(), 6))
print("Mean    :", round(test_probabilities.mean(), 6))
print("Median  :", round(np.median(test_probabilities), 6))


# ======================================================================
# 9. CREATE SUBMISSION FILE
# ======================================================================

print("\n" + "=" * 70)
print("CREATING KAGGLE SUBMISSION")
print("=" * 70)

submission = pd.DataFrame({
    "id": test["id"],
    TARGET: test_probabilities
})

submission_path = "stage9_submission.csv"

submission.to_csv(
    submission_path,
    index=False
)

print("\nSubmission created:")
print(submission_path)

print("\nSubmission shape:")
print(submission.shape)

print("\nSubmission columns:")
print(submission.columns.tolist())

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nPrediction range:")
print(
    submission[TARGET].min(),
    "to",
    submission[TARGET].max()
)


# ======================================================================
# 10. VALIDATE SUBMISSION FORMAT
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING SUBMISSION")
print("=" * 70)

assert list(submission.columns) == [
    "id",
    "Will_Buy_EV"
]

assert len(submission) == len(test)

assert submission["id"].equals(test["id"])

assert submission["Will_Buy_EV"].notna().all()

assert (
    (submission["Will_Buy_EV"] >= 0) &
    (submission["Will_Buy_EV"] <= 1)
).all()

print("Column check      : PASSED")
print("Row count check   : PASSED")
print("ID check          : PASSED")
print("Missing values    : PASSED")
print("Probability range : PASSED")


# ======================================================================
# STAGE 9 COMPLETE
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 9 COMPLETE")
print("=" * 70)

print("Final model       : Regularized_No_ID")
print("Training rows     :", len(X_final))
print("Final features    :", X_final.shape[1])
print("Test predictions  :", len(test_probabilities))
print("Submission file   :", submission_path)


print("=" * 70)

# ======================================================================
# STAGE 10 — FEATURE ENGINEERING VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 10 — FEATURE ENGINEERING VALIDATION")
print("=" * 70)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv("train.csv")

print("Train shape:", train.shape)


# ======================================================================
# 2. TARGET
# ======================================================================

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"

y = train[TARGET].map({
    "No": 0,
    "Yes": 1
})

X = train.drop(columns=[TARGET]).copy()

if ID_COLUMN in X.columns:
    X = X.drop(columns=[ID_COLUMN])

print("\nTarget distribution:")
print(train[TARGET].value_counts())


# ======================================================================
# 3. FEATURE ENGINEERING FUNCTION
# ======================================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------------
    # DIGIT DECOMPOSITION
    # --------------------------------------------------------------

    numeric_columns = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Number_of_Cars_Owned"
    ]

    for column in numeric_columns:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

            values = values.astype(int).abs()

            df[column + "_units"] = values % 10
            df[column + "_tens"] = (values // 10) % 10
            df[column + "_hundreds"] = (values // 100) % 10
            df[column + "_thousands"] = (values // 1000) % 10
            df[column + "_ten_thousands"] = (
                (values // 10000) % 10
            )

    # --------------------------------------------------------------
    # SELECTED INTERACTION FEATURES
    # --------------------------------------------------------------

    if {
        "Annual_Income_USD",
        "Subsidy_Available"
    }.issubset(df.columns):

        subsidy_numeric = (
            df["Subsidy_Available"]
            .map({"Yes": 1, "No": 0})
            .fillna(
                pd.to_numeric(
                    df["Subsidy_Available"],
                    errors="coerce"
                )
            )
            .fillna(0)
        )

        df["Subsidy_Income"] = (
            df["Annual_Income_USD"] *
            subsidy_numeric
        )

    if {
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work"
    }.issubset(df.columns):

        df["Total_Charging_Stations"] = (
            df["Charging_Stations_Near_Home"] +
            df["Charging_Stations_Near_Work"]
        )

        df["Charging_Station_Gap"] = (
            df["Charging_Stations_Near_Work"] -
            df["Charging_Stations_Near_Home"]
        )

    if {
        "Daily_Commute_km",
        "Range_Anxiety_Level"
    }.issubset(df.columns):

        range_anxiety_numeric = (
            df["Range_Anxiety_Level"]
            .map({
                "Low": 0,
                "Medium": 1,
                "High": 2
            })
        )

        if range_anxiety_numeric.notna().sum() == 0:
            range_anxiety_numeric = pd.to_numeric(
                df["Range_Anxiety_Level"],
                errors="coerce"
            )

        range_anxiety_numeric = (
            range_anxiety_numeric.fillna(0)
        )

        df["Range_Anxiety_Commute"] = (
            df["Daily_Commute_km"] *
            range_anxiety_numeric
        )

    return df


# ======================================================================
# 4. CREATE ENGINEERED FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("CREATING ENGINEERED FEATURES")
print("=" * 70)

X_engineered = create_features(X)

print("Original feature count :", X.shape[1])
print("Engineered feature count:", X_engineered.shape[1])

print("\nNew features:")
new_features = [
    column
    for column in X_engineered.columns
    if column not in X.columns
]

print(new_features)


# ======================================================================
# 5. CONSISTENT ENCODING
# ======================================================================

print("\n" + "=" * 70)
print("ENCODING FEATURES")
print("=" * 70)

categorical_columns = X_engineered.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

print("Categorical columns:")
print(categorical_columns)

X_encoded = pd.get_dummies(
    X_engineered,
    columns=categorical_columns,
    dummy_na=True
)

for column in X_encoded.columns:

    if X_encoded[column].dtype == bool:

        X_encoded[column] = (
            X_encoded[column].astype(int)
        )

X_encoded = X_encoded.replace(
    [np.inf, -np.inf],
    np.nan
)

for column in X_encoded.columns:

    if X_encoded[column].isna().any():

        median_value = X_encoded[column].median()

        if pd.isna(median_value):
            median_value = 0

        X_encoded[column] = (
            X_encoded[column].fillna(median_value)
        )

print("\nEncoded feature shape:")
print(X_encoded.shape)

print(
    "Missing values:",
    X_encoded.isna().sum().sum()
)


# ======================================================================
# 6. VALIDATION SPLIT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION SPLIT")
print("=" * 70)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training rows  :", len(X_train))
print("Validation rows:", len(X_valid))


# ======================================================================
# 7. CLASS BALANCING
# ======================================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\nScale positive weight:",
      round(scale_pos_weight, 4))


# ======================================================================
# 8. BASELINE MODEL
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING BASELINE MODEL")
print("=" * 70)

baseline_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_probabilities = (
    baseline_model.predict_proba(X_valid)[:, 1]
)

baseline_predictions = (
    baseline_probabilities >= 0.5
).astype(int)


# ======================================================================
# 9. EVALUATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 10 RESULTS")
print("=" * 70)

accuracy = accuracy_score(
    y_valid,
    baseline_predictions
)

precision = precision_score(
    y_valid,
    baseline_predictions,
    zero_division=0
)

recall = recall_score(
    y_valid,
    baseline_predictions,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    baseline_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_valid,
    baseline_probabilities
)

print("\nAccuracy :", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall   :", round(recall, 6))
print("F1 Score :", round(f1, 6))
print("ROC-AUC  :", round(roc_auc, 6))


# ======================================================================
# 10. COMPARISON WITH STAGE 8
# ======================================================================

stage8_auc = 0.941534

difference = roc_auc - stage8_auc

print("\n" + "=" * 70)
print("COMPARISON AGAINST STAGE 8")
print("=" * 70)

print("Stage 8 ROC-AUC :", round(stage8_auc, 6))
print("Stage 10 ROC-AUC:", round(roc_auc, 6))
print("Difference      :", f"{difference:+.6f}")


# ======================================================================
# 11. FEATURE IMPORTANCE
# ======================================================================

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": baseline_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.head(20).to_string(index=False)
)

importance.to_csv(
    "stage10_feature_importance.csv",
    index=False
)


# ======================================================================
# 12. SAVE RESULTS
# ======================================================================

results = pd.DataFrame({
    "Model": ["Stage10_Engineered_XGBoost"],
    "Features": [X_encoded.shape[1]],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1],
    "ROC_AUC": [roc_auc],
    "Stage8_ROC_AUC": [stage8_auc],
    "Difference": [difference]
})

results.to_csv(
    "stage10_model_comparison.csv",
    index=False
)


# ======================================================================
# STAGE 10 COMPLETE
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 10 COMPLETE")
print("=" * 70)

print("Stage 10 ROC-AUC:", round(roc_auc, 6))
print("Stage 8 ROC-AUC :", round(stage8_auc, 6))
print("Difference     :", f"{difference:+.6f}")

print("\nFiles created:")
print(" - stage10_model_comparison.csv")
print(" - stage10_feature_importance.csv")

print("\nDO NOT CREATE THE FINAL SUBMISSION FROM STAGE 10 YET.")
print("Review the results first.")

print("=" * 70)