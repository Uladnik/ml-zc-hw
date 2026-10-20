# %%
%load_ext autoreload
%autoreload 2
from src.mysrc_v1 import *

# %%
link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(link)
overview(uniform(df)).sort_values(by="missing")
# %%
y = df.msrp.values
X = df.drop("msrp", axis=1)
X.shape, y.shape
# %%
X.engine_fuel_type = X.engine_fuel_type.fillna(X.engine_fuel_type.mode()[0])
X.market_category = X.market_category.fillna("missing")
sns.histplot(X.engine_hp, bins=50)
# %%
X.engine_cylinders.value_counts().sort_index()
# %%
X.number_of_doors.value_counts().sort_index()
# %%
X.number_of_doors = X.number_of_doors.fillna(X.number_of_doors.mode()[0]).astype("str")
X.engine_cylinders = X.engine_cylinders.fillna(X.engine_cylinders.median()).astype(
    "str"
)
X.engine_hp = X.engine_hp.fillna(X.engine_hp.median())
overview(X).sort_values(by="missing")
# %%
X.year.min(), X.year.max()
# %%
X.year -= X.year.min()
# %%
sns.histplot(y, bins=50)
# %%
sns.histplot(np.log1p(y), bins=50)
# %%
y = np.log1p(y)
# %%
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
# %%
X_full_train.shape, X_test.shape, y_full_train.shape, y_test.shape
# %%
X_test.reset_index(drop=True, inplace=True)
X_full_train.reset_index(drop=True, inplace=True)
# %%
a = [10**i for i in range(-6, 6)]
pipeline = Pipeline([("preprocessor", preprocessor), (
    "model", RidgeCV(cv=10, alphas=a, scoring="r2")
)])
# %%
pipeline.fit(X_full_train, y_full_train)
# %%
get_feature_importance(pipeline)[:20]
# %%
get_analysis(pipeline)
# %%
evaluate_pipeline(pipeline, X_test, y_test)
