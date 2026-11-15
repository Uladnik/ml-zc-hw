# Import custom preprocessing functions and utilities
from preprocess import *

# 1. Load the raw dataset and perform an initial overview of missing values
csv = r"data/pancreatic_cancer_dataset.csv"
df = pd.read_csv(csv)

# 2. Standardize column names and drop the unique patient identifier
df = uniform(df).drop("patient_id", axis=1)

# 3. Handle missing values: fill missing categorical/text features with 'unknown'
summary = overview(df)
nan_cols = list(summary[summary["missing"] > 0].index)
for col in nan_cols:
    df[col] = df[col].fillna("unknown")

# 4. Convert diagnosis date into a numerical timestamp format for machine learning models
df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"]).astype("int64") // 10**9

# 5. Convert the target variable 'survived' into a binary format (1 / 0) and check class balance
df["survived"] = (df["survived"] == "yes").astype(int)

# 6. Perform stratified train/validation/test split to preserve class distribution
y = df["survived"]
df_full_train, df_test, y_full_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=1, stratify=y
)
df_train, df_val, y_train, y_val = train_test_split(
    df_full_train, y_full_train, test_size=0.25, random_state=1, stratify=y_full_train
)

# 7. Drop target variable, redundant year, and target-leaking features (Data Leakage prevention)
garbage = ["survived", "diagnosis_year", "five_year_status", "survival_months"]
df_full_train = df_full_train.drop(garbage, axis=1)
df_test = df_test.drop(garbage, axis=1)
df_train = df_train.drop(garbage, axis=1)
df_val = df_val.drop(garbage, axis=1)

# 8. Apply preprocessing pipeline (fitted strictly on train, then transformed on val, full_train, and test)
X_train = preprocessor.fit_transform(df_train)
X_val = preprocessor.transform(df_val)
X_full_train = preprocessor.transform(df_full_train)
X_test = preprocessor.transform(df_test)

# 9. Here's only the code for XGBoost.
# For all other models, see the corresponding ipynb file
xgb_params = {
    'n_estimators': 111,
    'learning_rate': 0.01271966900107984,
    'max_depth': 5,
    'subsample': 0.7337999287278185,
    'colsample_bytree': 0.8603499573440633
}

# Train and evaluate the winning model (XGBoost) on the holdout test set
xgb_model = XGBClassifier(**xgb_params, random_state=1)
xgb_model.fit(X_full_train, y_full_train)
y_pred = xgb_model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_pred[:, 1])

# 10. Save the final pipeline (preprocessor + best model) into a binary file
with open("model.bin", "wb") as f_out:
    pickle.dump((preprocessor, xgb_model), f_out)
    print('Model saved to model.bin file.')