import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(link)
df = uniform(df)
y = np.log1p(df.msrp.values)
X = df.drop("msrp", axis=1)

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=1)
eft = X_train.engine_fuel_type.mode()[0]
X_train.engine_fuel_type = X_train.engine_fuel_type.fillna(eft)
X_val.engine_fuel_type = X_val.engine_fuel_type.fillna(eft)
X_full_train.engine_fuel_type = X_full_train.engine_fuel_type.fillna(eft)
X_test.engine_fuel_type = X_test.engine_fuel_type.fillna(eft)
X_train.market_category = X_train.market_category.fillna("missing")
X_val.market_category = X_val.market_category.fillna("missing")
X_full_train.market_category = X_full_train.market_category.fillna("missing")
X_test.market_category = X_test.market_category.fillna("missing")
for col in ["number_of_doors", "engine_cylinders", "engine_hp"]:
    m = X_train[col].median()
    X_train[col] = X_train[col].fillna(m)
    X_val[col] = X_val[col].fillna(m)
    X_full_train[col] = X_full_train[col].fillna(m)
    X_test[col] = X_test[col].fillna(m)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_full_train = preprocessor.transform(X_full_train)
X_test = preprocessor.transform(X_test)

study = optuna.create_study(direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(lambda trial: objective(trial, 
    X_train, y_train, X_val, y_val,
    'xtr_r', ), n_trials=50)
model = ExtraTreesRegressor(**params, random_state=1)
model.fit(X_full_train, y_full_train)
y_pred = model.predict(X_test)
r2_score(y_test, y_pred)

with open('model.bin', 'wb') as f:
    pickle.dump((preprocessor, model), f)