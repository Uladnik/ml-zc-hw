# %%
from src.mysrc_v1 import *
# %%
link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(link)
# %%
overview(uniform(df))
# %%
y = (df.churn == "yes").astype(int)
X = df.drop(["churn", "customerid"], axis=1)
# %%
X.shape, y.shape
# %%
overview(X)
# %%
X.totalcharges = pd.to_numeric(X.totalcharges, errors="coerce")
# %%
X.totalcharges.isnull().sum()
# %%
X.loc[X.totalcharges.isnull(), ["tenure", "monthlycharges", "totalcharges"]]
# %%
X.totalcharges = X.totalcharges.fillna(0)
overview(X)[:10]
# %%
df.seniorcitizen = df.seniorcitizen.astype(str)
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
# %%
pipeline = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
# %%
pipeline.fit(X_train, y_train)
# %%
get_feature_importance(pipeline)[:15]
# %%
get_analysis(pipeline)
# %%
pipeline.named_steps["model"].C_
# %%
evaluate_pipeline(pipeline, X_test, y_test)
# %%
pipeline_mod = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(Cs=[2.7825594],
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
# %%
pipeline_mod.fit(X_train, y_train)
# %%
get_feature_importance(pipeline_mod)[:15]
# %%
get_analysis(pipeline_mod)
# %%
evaluate_pipeline(pipeline_mod, X_test, y_test)
# %%
pipeline_ideal = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(Cs=[0.0001],
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
# %%
pipeline_ideal.fit(X_train, y_train)
# %%
get_feature_importance(pipeline_ideal)[:15]
# %%
get_analysis(pipeline_ideal)
# %%
evaluate_pipeline(pipeline_ideal, X_test, y_test)
# %%
best_pipeline = pipeline_mod