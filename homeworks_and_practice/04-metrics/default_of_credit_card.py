# %%
from src.mysrc_v1 import *
# %%
dataset = uci(id=350)
df = dataset.data.features
y = dataset.data.targets.values
df.shape, y.shape
# %%
type(y), type(df)
# %%
y
# %%
overview(df).sort_values(by='missing')
# %%
df_full_train, df_test, y_full_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=1, stratify=y
)
df_full_train.shape, df_test.shape, y_full_train.shape, y_test.shape
# %%
pipeline = Pipeline([("preprocessor", preprocessor),
    ("model", LogisticRegressionCV(
        cv=1000, scoring="roc_auc", n_jobs=-1, random_state=1))])
# %%
pipeline.fit(df_full_train, y_full_train)
# %%
get_feature_importance(pipeline)[:15]
# %%
get_analysis(pipeline)
# %%
evaluate_pipeline(pipeline, df_test, y_test)