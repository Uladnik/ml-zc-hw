import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

link = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(link)
df = uniform(df)
y = df.fuel_efficiency_mpg.values
df = df.drop(columns=["fuel_efficiency_mpg"])

df_full_train, df_test, y_full_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=1)
df_full_train = add_columns(df_full_train)
df_test = add_columns(df_test)
for c in ('acceleration', 'horsepower'):
    med = df_full_train[c].median()
    df_full_train[c] = df_full_train[c].fillna(med)
    df_test[c] = df_test[c].fillna(med)
X_full_train = preprocessor.fit_transform(df_full_train)
X_test = preprocessor.transform(df_test)

study = optuna.create_study(direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
study.optimize(lambda trial: objective(trial, X_full_train, y_full_train,
    'lin_r', ), n_trials=50)
params = study.best_params
model = ElasticNet(**params, random_state=1)
model.fit(X_full_train, y_full_train)
with open('model.bin', 'wb') as f:
    pickle.dump((preprocessor, model), f)