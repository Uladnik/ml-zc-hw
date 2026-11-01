import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pickle
import seaborn as sns
import xgboost as xgb
from scipy import sparse as sp
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier, 
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    LogisticRegressionCV,
    RidgeCV,
)
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
from sklearn.model_selection import (
    cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, export_text
)


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
        [
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=True, handle_unknown="ignore", drop="first"
                ),
            )
        ]
    )
    preprocessor = ColumnTransformer([("num", nt, nums), ("cat", ct, cats)])
    return preprocessor


def evaluate_pipeline(pipeline, X_test, y_test):
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

    return None


def get_feature_importance(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": coefs}
        ).sort_values(by="importance", key=abs, ascending=False)
        return df_imp
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values(by="importance", ascending=False)
        return df_imp
    else:
        print("No coef_ attribute found in model")
        return None


def get_analysis(pipeline_or_search):
    if hasattr(pipeline_or_search, "best_estimator_"):
        best_est = pipeline_or_search.best_estimator_
        model = (best_est.named_steps.get("classifier") 
            or best_est.named_steps.get("regressor")
            or best_est.named_steps.get("model")
        )
        is_grid = True
        print("Best hyperparameters:", best_est.best_params_)
    else:
        model = (pipeline_or_search.named_steps.get("classifier") 
            or pipeline_or_search.named_steps.get("regressor")
            or pipeline_or_search.named_steps.get("model")
            if hasattr(pipeline_or_search, "named_steps")
            else pipeline_or_search
        )
        is_grid = False

    if model is None:
        print("Model step not found in pipeline.")
        return None

    if hasattr(model, "intercept_"):
        print("Intercept:", model.intercept_)
    if hasattr(model, "classes_"):
        print("Classes:", *model.classes_)
    if hasattr(model, "n_iter_"):
        print("Iterations:", model.n_iter_)
    if hasattr(model, "alpha_"):
        print("Selected alpha:", model.alpha_)
    if hasattr(model, "alphas_"):
        print("Tested alphas:", *model.alphas_)
    if hasattr(model, "C_"):
        print("Selected C:", model.C_)
    if hasattr(model, "n_estimators"):
        print("Number of estimators (trees):", model.n_estimators)
    if hasattr(model, "max_depth"):
        print("Max depth:", model.max_depth)
    if hasattr(model, "Cs_") and hasattr(model, "scores_"):
        print("\n--- CV Scores per C (Built-in CV) ---")
        print("C values:\tMean:\tStd:")
        class_key = list(model.scores_.keys())[0]
        scores_matrix = model.scores_[class_key]

        for Cs, scores_row in zip(model.Cs_, scores_matrix):
            print(
                round(float(Cs), 7),
                "\t",
                round(np.mean(scores_row), 7),
                "\t",
                round(np.std(scores_row), 7),
            )

    elif is_grid and hasattr(pipeline_or_search, "cv_results_"):
        print("\n--- GridSearchCV Results ---")
        cv_res = pipeline_or_search.cv_results_
        for mean_score, std_score, params in zip(
            cv_res["mean_test_score"], cv_res["std_test_score"], cv_res["params"]
        ):
            print(f"Params: {params} | Mean: {mean_score:.5f} | Std: {std_score:.5f}")
    elif (not is_grid
        and not hasattr(model, "alpha_")
        and not hasattr(model, "classes_")):
        print("\n(Standard model without built-in history)")

    return None


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

    columns = ['num_iter', 'train_res', 'val_res']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

def objective(trial, X_train, y_train, X_val=None, y_val=None, scoring='r2',
              cv=5, est=300, dep=50, spl=20, leaf=20, ft=1.0):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, est),
        'max_depth': trial.suggest_int('max_depth', 5, dep),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, spl),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, leaf),
        'max_features': trial.suggest_float('max_features', 0.3, ft),
        'random_state': 1,
    }
    
    model = RandomForestRegressor(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if scoring == 'r2':
            return r2_score(y_val, y_pred)
        if scoring == 'root_mean_squared_error':
            return np.sqrt(mean_squared_error(y_val, y_pred))
        if scoring == 'mean_absolute_error':
            return mean_absolute_error(y_val, y_pred)
        if scoring == 'neg_root_mean_squared_error':
            return -np.sqrt(mean_squared_error(y_val, y_pred))
        if scoring == 'neg_mean_absolute_error':
            return -mean_absolute_error(y_val, y_pred)
    else:
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1
        )
        return scores.mean()


preprocessor = preprocess()