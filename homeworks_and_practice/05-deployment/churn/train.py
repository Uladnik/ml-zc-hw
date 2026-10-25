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
df.seniorcitizen = df.seniorcitizen.astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
pipeline = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
pipeline.fit(X_train, y_train)

pipeline_mod = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(Cs=[2.7825594],
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
pipeline_mod.fit(X_train, y_train)

best_pipeline = pipeline_mod
with open('model_bin', "wb") as f_out:
    pickle.dump(best_pipeline, f_out)
    print("Saved model to model_bin")