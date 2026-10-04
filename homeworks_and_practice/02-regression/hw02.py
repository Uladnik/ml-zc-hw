# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv')

# %%
def overview(df):
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'nunique': df.nunique(),
        'missing': df.isnull().sum()
    })
    return summary

# %%
def shuffle_and_split(df, n1=0.6, n2=0.2, seed=42):
    n = len(df)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    
    df_shuffled = df.iloc[idx].reset_index(drop=True)
    
    n_train = int(n * n1)
    n_val = int(n * n2)
    
    df_train = df_shuffled[:n_train].reset_index(drop=True)
    df_valid = df_shuffled[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_shuffled[n_train + n_val:].reset_index(drop=True)
    
    return df_train, df_valid, df_test

# %%
def linear_regression(df, y, r=0):
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
def regularize(df_learn, y_learn, df_apply, y_apply):
    best_score = float('inf')
    best_r = 0.0000
    
    for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
        w0, w = linear_regression(df_learn, y_learn, r=r)
        y_pred = predict(df_apply, w0, w)
        score = round(rmse(y_apply, y_pred), 4)
        print(score, '\t', w0, '\t', r)
        
        if score < best_score:
            best_score = score
            best_r = r
            
    print(f"\nbest_r={best_r}, best_score={best_score}")
    return best_r

# %%
df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]
overview(df)
# Q1: horsepower 
# %%
df.horsepower.median()
# Q2: 254
# %%
sns.histplot(df.fuel_efficiency_mpg.values, bins=100)
# %%
df_0s = df.fillna(0)
overview(df_0s)
# %%
df_train_0s, df_valid_0s, df_test_0s = shuffle_and_split(df_0s)
df_train_0s.shape, df_valid_0s.shape, df_test_0s.shape
# %%
y_train = df_train_0s.fuel_efficiency_mpg.values
y_valid = df_valid_0s.fuel_efficiency_mpg.values
y_test = df_test_0s.fuel_efficiency_mpg.values
del df_train_0s['fuel_efficiency_mpg']
del df_valid_0s['fuel_efficiency_mpg']
del df_test_0s['fuel_efficiency_mpg']
# %%
df_train_0s.shape, df_valid_0s.shape, df_test_0s.shape
# %%
w0, w = linear_regression(df_train_0s, y_train)
y_pred = predict(df_valid_0s, w0, w)
score_0 = round(rmse(y_valid, y_pred), 3)
score_0
# %%
df_train_ms, df_valid_ms, df_test_ms = shuffle_and_split(df)
df_train_ms.shape, df_valid_ms.shape, df_test_ms.shape
# %%
del df_train_ms['fuel_efficiency_mpg']
del df_valid_ms['fuel_efficiency_mpg']
del df_test_ms['fuel_efficiency_mpg']
m = df_train_ms['horsepower'].mean()
df_train_ms = df_train_ms.fillna(m)
overview(df_train_ms)
# %%
df_valid_ms = df_valid_ms.fillna(m)
w0, w = linear_regression(df_train_ms, y_train)
y_pred = predict(df_valid_ms, w0, w)
score_m = round(rmse(y_valid, y_pred), 3)
score_0, score_m
# %%
# Q3: mean
# %%
best_r = regularize(df_train_0s, y_train, df_valid_0s, y_valid)
# Q4: 0
# %%
scores = []
for i in range(10):
    df_train_0s, df_valid_0s, df_test_0s = shuffle_and_split(df_0s, seed=i)
    y_train = df_train_0s.fuel_efficiency_mpg.values
    y_valid = df_valid_0s.fuel_efficiency_mpg.values
    y_test = df_test_0s.fuel_efficiency_mpg.values
    del df_train_0s['fuel_efficiency_mpg']
    del df_valid_0s['fuel_efficiency_mpg']
    del df_test_0s['fuel_efficiency_mpg']
    w0, w = linear_regression(df_train_0s, y_train)
    y_pred = predict(df_valid_0s, w0, w)
    score_0 = float(rmse(y_valid, y_pred))
    scores.append(score_0)
scores = np.array(scores)
scores
# %%
round(scores.std(), 3)
# Q5: 0.029
# %%
df_train_0s, df_valid_0s, df_test_0s = shuffle_and_split(df_0s, seed=9)
y_train = df_train_0s.fuel_efficiency_mpg.values
y_valid = df_valid_0s.fuel_efficiency_mpg.values
y_test = df_test_0s.fuel_efficiency_mpg.values
del df_train_0s['fuel_efficiency_mpg']
del df_valid_0s['fuel_efficiency_mpg']
del df_test_0s['fuel_efficiency_mpg']
df_full = pd.concat([df_train_0s, df_valid_0s], axis=0)
y_full = np.concatenate([y_train, y_valid])
w0, w = linear_regression(df_full, y_full, r=0.001)
y_pred = predict(df_test_0s, w0, w)
score_0 = round(rmse(y_test, y_pred), 3)
score_0
# Q6: 2.236
# %%
