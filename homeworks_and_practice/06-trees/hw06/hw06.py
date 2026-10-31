# %%time
%load_ext autoreload
%autoreload 2
import os

if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

# %%
link = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(link)
df = uniform(df)
overview(df)
# %%
df.acceleration = df.acceleration.fillna(0.0)
df.horsepower = df.horsepower.fillna(0.0)
# %%
sns.histplot(df.fuel_efficiency_mpg)
# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_full_train, y_train, y_val, y_test = (
    df_full_train.fuel_efficiency_mpg.values,
    df_train.fuel_efficiency_mpg.values,
    df_val.fuel_efficiency_mpg.values,
    df_test.fuel_efficiency_mpg.values,
)
del (
    df_full_train["fuel_efficiency_mpg"],
    df_train["fuel_efficiency_mpg"],
    df_val["fuel_efficiency_mpg"],
    df_test["fuel_efficiency_mpg"],
)

# %%
df_full_train.shape, df_train.shape, df_val.shape, df_test.shape, y_full_train.shape, y_train.shape, y_val.shape, y_test.shape
# %%
tree = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=1, max_depth=1)),
    ]
)

# %%
tree.fit(df_train, y_train)
# %%
fn = list(preprocessor.get_feature_names_out())
fn
# %%
print(export_text(tree.named_steps["model"], feature_names=fn))
# Q1: model_year
# %%
forest = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=1, n_estimators=10, n_jobs=-1)),

    ]
)
forest.fit(df_train, y_train)
y_pred = forest.predict(df_val)
root_mean_squared_error(y_val, y_pred)
# Q2: 1.837
# %%
for n in [10, 50, 100, 150]:
    forest = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=1, n_estimators=n, n_jobs=-1)),

        ]
    )
    forest.fit(df_train, y_train)
    y_pred = forest.predict(df_val)
    print(root_mean_squared_error(y_val, y_pred))
# Q3: 100
# %%
for d in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        forest = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(max_depth=d, random_state=1, n_estimators=n, n_jobs=-1)),
            ]
        )
        forest.fit(df_train, y_train)
        y_pred = forest.predict(df_val)
        scores.append(root_mean_squared_error(y_val, y_pred))
    print(sum(scores) / len(scores))
# Q4: 10
# %%
forest = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(max_depth=20, random_state=1, n_estimators=10, n_jobs=-1)),
    ]
)
forest.fit(df_train, y_train)
y_pred = forest.predict(df_val)
print(root_mean_squared_error(y_val, y_pred))
# %%
imps = []
for f, imp in zip(fn, forest.named_steps["model"].feature_importances_):
    imps.append((f, imp))
for f, imp in sorted(imps, key=lambda x: x[1], reverse=True):
    print(f, imp)
# Q5: vehicle_weight
# %%
dtrain = xgb.DMatrix(preprocessor.fit_transform(df_train), label=y_train, feature_names=fn)
dval = xgb.DMatrix(preprocessor.transform(df_val), label=y_val, feature_names=fn)
watchlist = [(dtrain, 'train'), (dval, 'val')]
eta_0_3 = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 12,
    'seed': 1,
    'verbosity': 1,
}
eta_0_1 = {
    'eta': 0.1, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 12,
    'seed': 1,
    'verbosity': 1,
}
# %%
%%capture output
model_0_3 = xgb.train(eta_0_3, dtrain, num_boost_round=100, evals=watchlist)
# %%
df_0_3 = parse_xgb_output(output)
# %%
%%capture output
model_0_1 = xgb.train(eta_0_1, dtrain, num_boost_round=100, evals=watchlist)
# %%
df_0_1 = parse_xgb_output(output)
# %%
plt.plot(df_0_3.num_iter, df_0_3.train_res, label='train 0.3')
plt.plot(df_0_3.num_iter, df_0_3.val_res, label='val 0.3')
plt.plot(df_0_1.num_iter, df_0_1.train_res, label='train 0.1')
plt.plot(df_0_1.num_iter, df_0_1.val_res, label='val 0.1')
plt.legend()
# %%
df_0_3.val_res.min(), df_0_1.val_res.min()
# Q6: 0.1