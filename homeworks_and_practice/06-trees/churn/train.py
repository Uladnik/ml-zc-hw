import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(link)
df = uniform(df)

y = (df.churn == "yes").astype(int)
X = df.drop(["churn", "customerid"], axis=1)

X.totalcharges = pd.to_numeric(X.totalcharges, errors="coerce")
X.totalcharges = X.totalcharges.fillna(0)
X.seniorcitizen = X.seniorcitizen.astype(str)

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, 
    random_state=1, stratify=y_full_train)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_full_train = preprocessor.transform(X_full_train)
X_test = preprocessor.transform(X_test)

study = optuna.create_study(direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(lambda trial: objective(trial, 
    X_train, y_train, X_val, y_val,
    'xgb_c', ), n_trials=50)
params = study.best_params

model = XGBClassifier(**params, random_state=1)
model.fit(X_full_train, y_full_train)
y_pred = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)
with open('model.bin', 'wb') as f:
    pickle.dump((preprocessor, model), f)