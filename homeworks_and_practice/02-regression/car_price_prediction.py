# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
)


# %%
def overview(df):
    summary = pd.DataFrame(
        {"dtype": df.dtypes, "nunique": df.nunique(), "missing": df.isnull().sum()}
    )
    return summary


# %%
def edit_columns(df, add=False, na_limit=0.05):
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    for col in df.columns:
        is_num = df[col].dtype.kind in ["i", "f", "u"]
        if (
            (add or is_num)
            and na_limit
            and df[col].isnull().sum() >= na_limit * len(df)
        ):
            df[f"{col}_is_na"] = df[col].isnull().astype(int)
        if not is_num:
            df[col] = df[col].astype("str").str.replace(" ", "_").str.lower()

    return df


# %%
def clean_features(df, categories=None, limit=30, full=True):
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
def shuffle_and_split(df, target, n1=0.6, n2=0.2, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)

    n = len(df)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx].reset_index(drop=True)
    features = list(df_shuffled.columns)

    n_train = int(n * n1)
    n_val = int(n * n2)

    y_train = df_shuffled[target].iloc[:n_train].reset_index(drop=True)
    y_valid = df_shuffled[target].iloc[n_train : n_train + n_val].reset_index(drop=True)
    y_test = df_shuffled[target].iloc[n_train + n_val :].reset_index(drop=True)
    features.remove(target)
    df_train = df_shuffled[features].iloc[:n_train].reset_index(drop=True)
    df_valid = (
        df_shuffled[features].iloc[n_train : n_train + n_val].reset_index(drop=True)
    )
    df_test = df_shuffled[features].iloc[n_train + n_val :].reset_index(drop=True)

    return df_train, df_valid, df_test, y_train, y_valid, y_test


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
# Special anchor for easy select of all the code above
# %%
df.shape
# %%
overview(df)
# %%
df = edit_columns(df)
overview(df)
# %%
sns.histplot(df["engine_hp"].values, bins=50)
# %%
df.engine_hp = df.engine_hp.fillna(df.engine_hp.median())
df.market_category = df.market_category.fillna("missing")
df.number_of_doors.value_counts()
# %%
df[df.number_of_doors == 2].msrp.median(), df[
    df.number_of_doors == 3
].msrp.median(), df[df.number_of_doors == 4].msrp.median()
# %%
df.number_of_doors = df.number_of_doors.fillna(df.number_of_doors.mode()[0]).astype(
    "str"
)
# %%
df.engine_cylinders.value_counts()
# %%
sns.histplot(df.engine_cylinders.values, bins=9)
# %%
for el in sorted(list(df.engine_cylinders.value_counts().index)):
    print(el, df[df.engine_cylinders == el].msrp.median())
# %%
df.engine_cylinders = df.engine_cylinders.fillna(df.engine_cylinders.mode()[0])
overview(df)
# %%
df.engine_fuel_type = df.engine_fuel_type.fillna(df.engine_fuel_type.mode()[0])
# %%
df.head()
df.year -= df.year.min()
df.head()
# %%
sns.histplot(df.msrp.values, bins=50)
# %%
sns.histplot(np.log1p(df.msrp.values), bins=50)
# %%
df_train, df_valid, df_test, y_train, y_valid, y_test = shuffle_and_split(
    df, "msrp", seed=2
)
df_train.shape, df_valid.shape, df_test.shape, y_train.shape, y_valid.shape, y_test.shape
# %%
y_train, y_valid, y_test = np.log1p(y_train), np.log1p(y_valid), np.log1p(y_test)
y_train.head()
# %%
X_train, categories = clean_features(df_train, limit=0)
X_valid = clean_features(df_valid, categories, limit=0)
X_test = clean_features(df_test, categories, limit=0)
# %%
X_train.shape, X_valid.shape, X_test.shape
# %%
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# %%
best_r = regularize(X_train, y_train, X_valid, y_valid)
# %%
X_full = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
y_full = np.concatenate([y_train, y_valid])
# %%
w0, w = linear_regression(X_full, y_full, r=best_r)
# %%
y_pred = predict(X_test, w0, w)
rmse(y_test, y_pred)
# %%
sns.histplot(y_test, color="blue", alpha=0.5, bins=50)
sns.histplot(y_pred, color="red", alpha=0.5, bins=50)
# %%
car = df_test.iloc[22].to_dict()
car = pd.DataFrame([car])
car
# %%
car = clean_features(car, categories, full=True)
car.shape
# %%
car = car.reindex(columns=X_full.columns, fill_value=0)
car.shape
# %%
y_pred = predict(car, w0, w)
y_pred
# %%
np.expm1(y_pred)
# %%
print(f'Target: {np.expm1(y_test.iloc[22])}')
print(f'Prediction: {np.expm1(y_pred)}')