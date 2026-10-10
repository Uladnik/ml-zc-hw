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
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
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
def get_X(df, encoder=None, scaler=None, freq=0, full=True, sparse=False):
    is_train = encoder is None and scaler is None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    if num_cols:
        if is_train:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(df[num_cols])
        else:
            X_num = scaler.transform(df[num_cols])
    else:
        X_num = None
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
        return X, encoder, scaler
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
df.shape
# %%
overview(uniform(df))
# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_valid = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_full_train, y_full_train = get_Y(df_full_train, target="msrp")
df_train, y_train = get_Y(df_train, target="msrp")
df_valid, y_valid = get_Y(df_valid, target="msrp")
df_test, y_test = get_Y(df_test, target="msrp")
df_train.shape, df_valid.shape, df_test.shape, y_train.shape, y_valid.shape, y_test.shape
# %%
sns.histplot(y_train, bins=50)
# %%
sns.histplot(np.log1p(y_train), bins=50)
# %%
overview(df_full_train)
# %%
sns.histplot(df_full_train.engine_hp.values, bins=50)
# %%
for df in [df_train, df_valid, df_test, df_full_train]:
    df.year -= df.year.min()
    df.engine_cylinders = df.engine_cylinders.fillna(df.engine_cylinders.mode()[0])
    df.engine_fuel_type = df.engine_fuel_type.fillna(df.engine_fuel_type.mode()[0])
    df.number_of_doors = df.number_of_doors.fillna(df.number_of_doors.mode()[0])
    df.engine_hp = df.engine_hp.fillna(df.engine_hp.median())
    df.market_category = df.market_category.fillna("missing")
# %%
y_train = np.log1p(y_train)
y_valid = np.log1p(y_valid)
y_test = np.log1p(y_test)
y_full_train = np.log1p(y_full_train)
# %%
features = mutual_impact("ridge", df_full_train, y_full_train)
features
# %%
overview(df_full_train)
# %%
for df in [df_train, df_valid, df_test, df_full_train]:
    df.engine_cylinders = df.engine_cylinders.astype("str")
    df.number_of_doors = df.number_of_doors.astype("str")
# %%
X_train, encoder, scaler = get_X(df_train)
X_valid = get_X(df_valid, encoder=encoder, scaler=scaler)
X_test = get_X(df_test, encoder=encoder, scaler=scaler)
X_train.shape, X_valid.shape, X_test.shape
# %%
model = evaluate_model("ridge", X_train, y_train, X_valid, y_valid)["model"]
# %%
X_full_train, encoder, scaler = get_X(df_full_train)
X_test = get_X(df_test, encoder=encoder, scaler=scaler)
X_full_train.shape, X_test.shape
# %%
model.fit(X_full_train, y_full_train)
y_pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))