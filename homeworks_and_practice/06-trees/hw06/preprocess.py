import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pickle
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from scipy import sparse as sp
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor, 
    RandomForestClassifier, 
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
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


def line(trial, X, y, mod='r', scor_r='r2', scor_c='roc_auc', cv=5):
    l1 = trial.suggest_float('l1_ratio', 0.0, 1.0)
    if mod == 'r':
        alpha = trial.suggest_float('alpha', 1e-6, 1e4, log=True)
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1, max_iter=10000,  random_state=1)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_r, n_jobs=-1)
    elif mod == 'c':
        C = trial.suggest_float('C', 1e-6, 1e4, log=True)
        model = LogisticRegression(solver='saga', penalty='elasticnet',        
            C=C, l1_ratio=l1, max_iter=10000, random_state=1)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_c, n_jobs=-1)
    else:
        raise ValueError("""
Invalid model type. Choose mod:
'r' for regression or 'c' for classification
""")            
    return scores.mean()


def forest(trial, X, y, mod='r', scor_r='r2', scor_c='roc_auc',
              cv=5, est=300, dep=50, spl=20, leaf=20, ft=None):
    if ft is None:
        ft = X.shape[1]
    ft_low = int(X.shape[1] * 0.3)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, est),
        'max_depth': trial.suggest_int('max_depth', 5, dep),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, spl),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, leaf),
        'max_features': trial.suggest_int('max_features', ft_low, ft),
        'random_state': 1,
    }
    
    if mod == 'r':
        model = RandomForestRegressor(**params)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_r, n_jobs=-1)
    elif mod == 'c':
        model = RandomForestClassifier(**params)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_c, n_jobs=-1)
    elif mod == 'xr':
        model = ExtraTreesRegressor(**params)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_r, n_jobs=-1)
    elif mod == 'xc':
        model = ExtraTreesClassifier(**params)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scor_c, n_jobs=-1)
    else:
        raise ValueError("""
Invalid model type. Choose mod:
'r' for regression
'c' for classification
'xr' for extra trees regression
'xc' for extra trees classification
""")
    return scores.mean()


def catboost(trial, X, y, mod='r', scor_r='r2', scor_c='roc_auc', cv=5):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 8), 
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
        'random_seed': 1,
        'verbose': 0,
    }
    
    if mod == 'r':
        model = CatBoostRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_r, n_jobs=-1)
        
    elif mod == 'c':
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_c, n_jobs=-1)
        
    else:
        raise ValueError(
            "Choose 'r' for regression or 'c' for classification.")
    return scores.mean()


def objective_xgboost(trial, X, y, mod='r', scor_r='r2', scor_c='roc_auc', cv=5):
    # Гиперпараметры для XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': 1,
        'n_jobs': -1,
        'verbosity': 0  # Отключаем лишний вывод логов в консоль
    }
    
    if mod == 'r':
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_r, n_jobs=1)
        
    elif mod == 'c':
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_c, n_jobs=1)
        
    else:
        raise ValueError("Choose 'r' for regression or 'c' for classification.")
        
    return scores.mean()


def objective_lightgbm(trial, X, y, mod='r', scor_r='r2', scor_c='roc_auc', cv=5):
    # Гиперпараметры для LightGBM
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63), # Важный параметр для leaf-wise роста
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Доля объектов (bagging_fraction)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Доля признаков (feature_fraction)
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), # Минимальное число объектов в листе
        'random_state': 1,
        'n_jobs': -1,
        'verbose': -1  # Отключаем лишний вывод и предупреждения в консоль
    }
    
    if mod == 'r':
        model = LGBMRegressor(**params)
        # По аналогии с XGBoost: распараллеливание внутри модели, кросс-валидация в 1 поток
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_r, n_jobs=1)
        
    elif mod == 'c':
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scor_c, n_jobs=1)
        
    else:
        raise ValueError("Choose 'r' for regression or 'c' for classification.")
        
    return scores.mean()


def objective(trial, X, y, mod, scor_r='r2', scor_c='roc_auc', *,
              seed=1, cv=5, vrb=0, row=1.0, col=1.0,
              est=250, rate=0.3, dep=8, spl=20, leaf=20):
              
    lin_params = {'max_iter': 10000, 'random_state': seed,
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
    }
    for_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, est),
        'max_depth': trial.suggest_int('max_depth', 5, dep*3),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, spl),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, leaf),
        'max_features': trial.suggest_float('max_features', 0.3, col),
        'random_state': seed,
        'n_jobs': -1,
    }
    boost_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, est*4),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, rate, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, dep),
        'subsample': trial.suggest_float('subsample', 0.3, row),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, col),
        'random_state': seed,
    }
              
    mods = {
        'lin_c': (LogisticRegression, {**lin_params,
            'penalty': 'elasticnet', 'solver': 'saga',
            'C': trial.suggest_float('C', 1e-6, 1e4, log=True),
        }),
        'lin_r': (ElasticNet, {**lin_params,
            'alpha': trial.suggest_float('alpha', 1e-6, 1e4, log=True),
        }),
        'for_c': (RandomForestClassifier, for_params),
        'for_r': (RandomForestRegressor, for_params),
        'xtr_c': (ExtraTreesClassifier, for_params),
        'xtr_r': (ExtraTreesRegressor, for_params),
        'xgb_c': (XGBClassifier, {
            **boost_params, 'n_jobs': -1, 'verbosity': vrb}),
        'xgb_r': (XGBRegressor, {
            **boost_params, 'n_jobs': -1, 'verbosity': vrb}),
        'lgb_c': (LGBMClassifier, {
            **boost_params, 'n_jobs': -1, 'verbose': vrb-1}),
        'lgb_r': (LGBMRegressor, {
            **boost_params, 'n_jobs': -1, 'verbose': vrb-1}),
        'cgb_c': (CatBoostClassifier, {
            **boost_params, 'thread_count': -1, 'verbose': vrb}),
        'cgb_r': (CatBoostRegressor, {
            **boost_params, 'thread_count': -1, 'verbose': vrb})
    }

    if mod not in mods:
        raise ValueError(
            'Invalid model type. Choose from: {}'.format(mods.keys())
        )
    cvj = -1 if 'lin' in mod else 1
    scoring = scor_c if '_c' in mod else scor_r

    model_class, model_params = mods[mod]
    model = model_class(**model_params)
    score = cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=cvj).mean()

    return score

            
preprocessor = preprocess()