# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from scipy import sparse as sp
from IPython.display import display

# %%
df = pd.read_csv(
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
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
                sparse_output=sparse
            )
            X_cat = encoder.fit_transform(df[cat_cols])
        else:
            X_cat = encoder.transform(df[cat_cols])
            
    if X_num is not None and X_cat is not None:
        if sparse:
            X = sp.hstack([X_num, X_cat])
        else:
            if sp.issparse(X_num): X_num = X_num.toarray()
            if sp.issparse(X_cat): X_cat = X_cat.toarray()
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
def mutual_impact(df, y):
    df = df.copy()
    cats = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in cats:
        df[col] = df[col].astype("category").cat.codes

    mi = mutual_info_classif(df, y, discrete_features=df.columns.isin(cats))
    return pd.Series(mi, index=df.columns).sort_values(ascending=False)
# %%
# Special anchor for easy select of all the code above
# %%
overview(uniform(df))
# %%
df.shape
# %%
del df['customerid']
# %%
for col in df.columns:
    print(col, df[col].unique()[:10].tolist())
# %%
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges.isnull().sum()
# %%
df[df.totalcharges.isnull()][["tenure", "monthlycharges", "totalcharges"]]
# %%
df.totalcharges = df.totalcharges.fillna(0)
# %%
df.seniorcitizen = df.seniorcitizen.astype(str)
# %%
df.churn = (df.churn == "yes").astype(int)
df.churn.head()
# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_valid = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train.shape, df_valid.shape, df_test.shape
# %%
df_train.head().T
# %%
features = mutual_impact(df_full_train, df_full_train.churn)
features
# %%
del features["churn"]
# %%
useful = features[features > 0.02].sort_values(ascending=False).index.tolist()
useful
# %%
df_train, y_train = get_Y(df_train, target="churn")
df_valid, y_valid = get_Y(df_valid, target="churn")
df_test, y_test = get_Y(df_test, target="churn")
# %%
X_train, encoder, scaler = get_X(df_train, freq=50)
X_valid = get_X(df_valid, encoder=encoder, scaler=scaler)
X_test = get_X(df_test, encoder=encoder, scaler=scaler)
X_train.shape, X_valid.shape, X_test.shape
# %%
default_model = evaluate_model("log", X_train, y_train, X_valid, y_valid)["model"]
# %%
X_train, encoder, scaler = get_X(df_train[useful], freq=50)
X_valid = get_X(df_valid[useful], encoder=encoder, scaler=scaler)
X_test = get_X(df_test[useful], encoder=encoder, scaler=scaler)
X_train.shape, X_valid.shape, X_test.shape
# %%
cut_model = evaluate_model("log", X_train, y_train, X_valid, y_valid)["model"]
# %%
df_full_train, y_full_train = get_Y(df_full_train, target="churn")
# %%
df_full_train.head().T
# %%
X_full_train, encoder, scaler = get_X(df_full_train, freq=50)
X_test = get_X(df_test, encoder=encoder, scaler=scaler)
X_full_train.shape, X_test.shape
# %%
default_model.fit(X_full_train, y_full_train)
y_pred = default_model.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
X_full_train, encoder, scaler = get_X(df_full_train[useful], freq=50)
X_test = get_X(df_test[useful], encoder=encoder, scaler=scaler)
X_full_train.shape, X_test.shape
# %%
cut_model.fit(X_full_train, y_full_train)
y_pred = cut_model.predict(X_test)
accuracy_score(y_test, y_pred)