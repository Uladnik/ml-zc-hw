import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier
from scipy import sparse as sp
from ucimlrepo import fetch_ucirepo as uci


def overview(df):
    summary = pd.DataFrame(
        {"dtype": df.dtypes, "nunique": df.nunique(), "missing": df.isnull().sum()}
    ).join(df.head().T)
    return summary


def uniform(df):
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].astype("str").str.replace(" ", "_").str.lower()
    return df


def add_columns(df, na_limit=0.05):
    df = df.copy()

    for col in df.select_dtypes(include=["number"]).columns:
        if na_limit and df[col].isnull().sum() >= na_limit * len(df):
            df[f"{col}_is_na"] = df[col].isnull().astype(int)

    return df


def preprocess():
    nums = make_column_selector(dtype_include=["number"])
    cats = make_column_selector(dtype_exclude=["number"])
    nt = Pipeline([("scaler", StandardScaler())])
    ct = Pipeline(
        [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer([("num", nt, nums), ("cat", ct, cats)])
    return preprocessor


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if is_classifier(pipeline):
        print("--- Classification Report ---")
        print(classification_report(y_test, y_pred))
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC Score: {auc}")
    else:
        print("R2 Score:", r2_score(y_test, y_pred))
        print("RMSE:", root_mean_squared_error(y_test, y_pred))

    return pipeline


def get_feature_importance(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": coefs}
        ).sort_values(by="importance", key=abs, ascending=False)
        return df_imp
    else:
        print("No coef_ attribute found in model")
        return None


def get_analysis(pipeline):
    model = pipeline.named_steps["model"]

    if hasattr(model, "intercept_"):
        print(f"Intercept: {model.intercept_}")
    if hasattr(model, "classes_"):
        print(f"Classes: {model.classes_}")
    if hasattr(model, "n_iter_"):
        print(f"Iterations: {model.n_iter_}")
    if hasattr(model, "Cs_"):
        print(f"C values: {model.Cs_}")
    if hasattr(model, "scores_"):
        print(f"Scores: {model.scores_}")
        return model.scores_
    else:
        print("No scores_ attribute found in model")
        return None


preprocessor = preprocess()
