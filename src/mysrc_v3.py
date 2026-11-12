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
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor, 
    RandomForestClassifier, 
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import (
    cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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
    ct = Pipeline([("onehot", OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="first"
                 ))])
    preprocessor = ColumnTransformer([("num", nt, nums), ("cat", ct, cats)])
    return preprocessor


def objective(trial, X, y, mod, scor_r='r2', scor_c='roc_auc', *,
              seed=1, cv=5, vrb=0, row=1.0, col=1.0,
              est=250, rate=0.3, dep=8, spl=20, leaf=20):
              
    mods = ('lin_r', 'lin_c', 'for_r', 'for_c', 'xtr_r', 'xtr_c',
            'cgb_r', 'cgb_c', 'lgb_r', 'lgb_c', 'xgb_r', 'xgb_c')              

    if mod not in mods:
        raise ValueError('Invalid model type. Choose from:', mods)
    elif mod.startswith('lin'):              
        lin_params = {'max_iter': 10000, 'random_state': seed,
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)}
        if mod == 'lin_r':
            model = ElasticNet(**lin_params,
                alpha=trial.suggest_float('alpha', 1e-6, 1e4, log=True))
        elif mod == 'lin_c':
            model = LogisticRegression(**lin_params,
                penalty='elasticnet', solver='saga',
                C=trial.suggest_float('C', 1e-6, 1e4, log=True))
    elif mod.startswith('for') or mod.startswith('xtr'):
        for_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, est),
            'max_depth': trial.suggest_int('max_depth', 5, dep*3),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, spl),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, leaf),
            'max_features': trial.suggest_float('max_features', 0.3, col),
            'random_state': seed, 'n_jobs': -1,
        }
        if mod == 'for_r': model = RandomForestRegressor(**for_params)
        elif mod == 'for_c': model = RandomForestClassifier(**for_params)
        elif mod == 'xtr_r': model = ExtraTreesRegressor(**for_params)
        elif mod == 'xtr_c': model = ExtraTreesClassifier(**for_params)
    elif mod.startswith('cgb'):
        gb_params = {
            'iterations': trial.suggest_int('iterations', 100, est*4),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, rate, log=True),
            'depth': trial.suggest_int('depth', 3, dep),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.3, row),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, col),
            'random_state': seed, 'verbose': vrb, 'thread_count': -1
        }
        if mod == 'cgb_r': model = CatBoostRegressor(**gb_params)
        elif mod == 'cgb_c': model = CatBoostClassifier(**gb_params)
    else:
        gb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, est*4),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, rate, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, dep),
            'subsample': trial.suggest_float('subsample', 0.3, row),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, col),
            'random_state': seed, 'n_jobs': -1
        }
        if mod == 'lgb_c': model = LGBMClassifier(**gb_params, verbose=vrb-1)
        elif mod == 'lgb_r': model = LGBMRegressor(**gb_params, verbose=vrb-1)
        elif mod == 'xgb_c': model = XGBClassifier(**gb_params, verbosity=vrb)
        elif mod == 'xgb_r': model = XGBRegressor(**gb_params, verbosity=vrb)
    
    cvj = -1 if mod.startswith('lin') else 1
    scoring = scor_c if mod.endswith('_c') else scor_r
    score = cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=cvj).mean()

    return score

            
preprocessor = preprocess()