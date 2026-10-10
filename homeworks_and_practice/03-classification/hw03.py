# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from scipy import sparse as sp

# %%
df = pd.read_csv(
    'https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv'
)


# %%
def overview(df):
    summary = pd.DataFrame(
        {"dtype": df.dtypes, "nunique": df.nunique(), "missing": df.isnull().sum()}
    ).join(df.head().T)
    return summary


# %%
def uniform(df):
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].astype("str").str.replace(" ", "_").str.lower()
    return df


# %%
def add_columns(df, na_limit=0.05):
    df = df.copy()

    for col in df.select_dtypes(include=["number"]).columns:
        if na_limit and df[col].isnull().sum() >= na_limit * len(df):
            df[f"{col}_is_na"] = df[col].isnull().astype(int)

    return df


# %%
def get_X(df, encoder=None, #scaler=None,
           freq=0, full=True, sparse=False):
    is_train = encoder is None #and scaler is None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # if num_cols:
    #     if is_train:
    #         scaler = StandardScaler()
    #         X_num = scaler.fit_transform(df[num_cols])
    #     else:
    #         X_num = scaler.transform(df[num_cols])
    # else:
    #     X_num = None
    if num_cols:
        X_num = df[num_cols].values
    else:
        X_num = None
    #
    if sparse and X_num is not None:
        X_num = sp.csr_matrix(X_num)

    X_cat = None
    if full and cat_cols:
        if is_train:
            encoder = OneHotEncoder(
                min_frequency=(freq if freq > 0 else None),
                handle_unknown="ignore",
                sparse_output=sparse,
            )
            X_cat = encoder.fit_transform(df[cat_cols])
        else:
            X_cat = encoder.transform(df[cat_cols])

    if X_num is not None and X_cat is not None:
        if sparse:
            X = sp.hstack([X_num, X_cat])
        else:
            if sp.issparse(X_num):
                X_num = X_num.toarray()
            if sp.issparse(X_cat):
                X_cat = X_cat.toarray()
            X = np.hstack([X_num, X_cat])
    elif X_num is not None:
        X = X_num
    elif X_cat is not None:
        X = X_cat
    else:
        X = None

    if is_train:
        return X, encoder#, scaler
    else:
        return X


# %%
def get_Y(df, target):
    df = df.copy()
    y = df[target].values
    del df[target]
    return df.reset_index(drop=True), y


# %%
def evaluate_model(func, X_train, y_train, X_val, y_val, low=-5, high=4, iter=10**9):
    params_list = [10.0**i for i in range(low, high + 1)]

    if func.lower() == "ridge":
        best_rmse = float("inf")
        best_alpha = None
        best_model = None

        for alpha in params_list:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            if best_model is None or round(rmse, 4) < round(best_rmse, 4):
                best_rmse = rmse
                best_alpha = alpha
                best_model = model

            print(rmse, "\t", model.intercept_, "\t", alpha)
        return {"model": best_model, "score": best_rmse, "param": best_alpha}

    elif func.lower() in ["log", "logistic"]:
        best_acc = -1.0
        best_c = None
        best_model = None

        for C in reversed(params_list):
            model = LogisticRegression(C=C, max_iter=iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            if round(acc, 4) > round(best_acc, 4):
                best_acc = acc
                best_c = C
                best_model = model

            print(acc, "\t", model.intercept_[0], "\t", C)
        return {"model": best_model, "score": best_acc, "param": best_c}

    else:
        raise ValueError(
            "Invalid model type (func). Choose either 'ridge' or 'log'/'logistic'."
        )


# %%
def mutual_impact(func, df, y):
    df = df.copy()
    cats = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in cats:
        df[col] = df[col].astype("category").cat.codes

    if func.lower() in ("log", "logistic"):
        mi = mutual_info_classif(df, y, discrete_features=df.columns.isin(cats))
    elif func.lower() == "ridge":
        mi = mutual_info_regression(df, y, discrete_features=df.columns.isin(cats))
    else:
        raise ValueError(
            "Invalid model type (func). Choose either 'ridge' or 'log'/'logistic'."
        )
    return pd.Series(mi, index=df.columns).sort_values(ascending=False)


# %%
# Special anchor for easy select of all the code above
# %%
df.shape
# %%
overview(uniform(df))
# %%
df.lead_score = df.lead_score.fillna(0.0)
df.annual_income = df.annual_income.fillna(0.0)
for col in df.select_dtypes(exclude=['number']).columns:
    df[col] = df[col].fillna('NA')
# %%
overview(df)
# %%
df.industry.mode()[0]
# Q1: technology
# %%
df.converted.mean()
# %%
df.select_dtypes(include=['number']).corr()
# Q2: #1
# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)
# %%
cats = df_train.select_dtypes(exclude=['number']).columns.tolist()
mutual_impact('log', df_train[cats], df_train.converted).round(2)
# Q3: lead_source
# %%
df_full_train, y_full_train = get_Y(df_full_train, 'converted')
df_train, y_train = get_Y(df_train, 'converted')
df_val, y_val = get_Y(df_val, 'converted')
df_test, y_test = get_Y(df_test, 'converted')
df_train.shape, df_val.shape, df_test.shape
# %%
X_train, encoder = get_X(df_train)
X_val = get_X(df_val, encoder=encoder)
X_test = get_X(df_test, encoder=encoder)
# %%
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
base = accuracy_score(y_val, y_pred)
print(round(base, 2))
# Q4: 0.65
# %%
for col in df_train.columns:
    m1 = df_train.columns.tolist()
    m1.remove(col)
    X_train, encoder = get_X(df_train[m1])
    X_val = get_X(df_val[m1], encoder=encoder)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(abs(acc - base), '\t', col)
# Q5: lead_source
# %%
df_train.columns, df_val.columns
# %%
X_train, encoder = get_X(df_train)
X_val = get_X(df_val, encoder=encoder)
# %%
evaluate_model('log', X_train, y_train, X_val, y_val, low=-6, high=-3)['param']
# Q6: 0.001