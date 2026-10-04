# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)


# %%
def overview(df):
    summary = pd.DataFrame(
        {"dtype": df.dtypes, "nunique": df.nunique(), "missing": df.isnull().sum()}
    )
    return summary


# %%
# fmt: off
def edit_columns(df, nfill="median", sfill="missing", na_limit=0.05, nunique=10):
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    for col in df.select_dtypes(include=["number"]).columns:
        if na_limit and df[col].isnull().mean() >= na_limit:
            df[f"{col}_is_na"] = df[col].isnull().astype(int)
        if nunique and df[col].nunique() <= nunique:
            if   sfill == "missing": val = 'missing'
            elif sfill == "mode":    val = df[col].mode()[0]
            else:                    val = sfill
            df[col] = df[col].fillna(val).astype("str")
        else:
            if   nfill == "median": val = df[col].median()
            elif nfill == "mean":   val = df[col].mean()
            elif nfill == "mode":   val = df[col].mode()[0]
            elif nfill == "zero":   val = 0
            else:                   val = nfill
            df[col] = df[col].fillna(val)

    for col in df.select_dtypes(exclude=["number"]).columns:
        df[col] = df[col].astype("str").str.replace(" ", "_").str.lower()
        if   sfill == "missing": val = 'missing'
        elif sfill == "mode":    val = df[col].mode()[0]
        else:                    val = sfill
        df[col] = df[col].fillna(val)

    return df
# fmt: on


# %%
def clean_features(df, categories=None, limit=30, full=False):
    df = df.copy()
    is_train = categories is None
    if is_train:
        categories = dict()

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    df_cleaned = df[num_cols]

    if full:
        for col in cat_cols:
            if is_train:
                n = df[col].value_counts()
                top = n[n > limit].index
                categories[col] = top
            else:
                top = categories[col]

            df[col] = df[col].where(df[col].isin(top), "other")
            binary = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df_cleaned = pd.concat([df_cleaned, binary], axis=1)

    if is_train:
        return df_cleaned, categories
    else:
        return df_cleaned


# %%
def shuffle_and_split(df, n1=0.6, n2=0.2, seed=2):
    n = len(df)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx].reset_index(drop=True)

    n_train = int(n * n1)
    n_val = int(n * n2)

    df_train = df_shuffled[:n_train].reset_index(drop=True)
    df_valid = df_shuffled[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df_shuffled[n_train + n_val :].reset_index(drop=True)

    return df_train, df_valid, df_test


# %%
def linear_regression(df, y, r=0.000001):
    X = df.values
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X) + r * np.eye(X.shape[1])
    w_full = np.linalg.inv(XTX).dot(X.T).dot(y)

    w0 = w_full[0]
    w = np.array(w_full[1:])

    return w0, w


# %%
def predict(df, w0, w):
    return w0 + df.values.dot(w)


# %%
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# %%
def regularize(df_learn, y_learn, df_apply, y_apply, j=-6, k=2):
    best_score = float("inf")
    best_r = 0.000001

    for i in range(j, k):
        r_val = 10**i
        w0, w = linear_regression(df_learn, y_learn, r=r_val)
        y_pred = predict(df_apply, w0, w)
        score = rmse(y_apply, y_pred)
        print(score, "\t", w0, "\t", r_val)

        if round(score, 4) <= round(best_score, 4):
            best_score = score
            best_r = r_val

    print(f"\nbest_r={best_r}, best_score={best_score}")
    return best_r


# %%
# This function wasn't used in the project
def get_target(df):
    features = list(df.columns)
    return df[features[:-1]], df[features[-1]]


# %%
overview(df)
# %%
sns.histplot(df.horsepower.values, bins=50)
# %%
sns.histplot(df.acceleration.values, bins=50)
# %%
print([df[c].unique() for c in df.columns if df[c].nunique() <= 10])
# %%
df = edit_columns(df, nfill="mean", sfill="mode")
overview(df)
# %%
y = df.fuel_efficiency_mpg
del df["fuel_efficiency_mpg"]
df = pd.concat([df, y], axis=1)
overview(df)
# %%
n = np.random.randint(0, len(df), 1)
n
# %%
df_train, df_valid, df_test = shuffle_and_split(df, seed=n, n1=0.7, n2=0.15)
df_train.shape, df_valid.shape, df_test.shape
# %%
df_train, y_train = get_target(df_train)
df_valid, y_valid = get_target(df_valid)
df_test, y_test = get_target(df_test)
df_train.shape, df_valid.shape, df_test.shape
# %%
sns.histplot(y_train, bins=50)
# %%
X_train, categories = clean_features(df_train, full=True, limit=0)
X_valid = clean_features(df_valid, categories, full=True, limit=0)
X_test = clean_features(df_test, categories, full=True, limit=0)
X_train.shape, X_valid.shape, X_test.shape
# %%
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# %%
best_r = regularize(X_train, y_train, X_valid, y_valid)
# %%
X_full = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
y_full = np.concatenate([y_train, y_valid])
w0, w = linear_regression(X_full, y_full, r=best_r)
y_pred = predict(X_test, w0, w)
rmse(y_test, y_pred)
# %%
sns.histplot(y_test, color="blue", alpha=0.5, bins=50)
sns.histplot(y_pred, color="red", alpha=0.5, bins=50)