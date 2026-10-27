import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(link)
df = uniform(df)

y = np.log1p(df.msrp.values)
X = df.drop("msrp", axis=1)

X.engine_fuel_type = X.engine_fuel_type.fillna(X.engine_fuel_type.mode()[0])
X.market_category = X.market_category.fillna("missing")
X.number_of_doors = X.number_of_doors.fillna(X.number_of_doors.mode()[0]).astype("str")
X.engine_cylinders = X.engine_cylinders.fillna(X.engine_cylinders.median()).astype("str")
X.engine_hp = X.engine_hp.fillna(X.engine_hp.median())
X.year -= X.year.min()

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

a = [10**i for i in range(-6, 6)]
pipeline = Pipeline([("preprocessor", preprocessor), (
    "model", RidgeCV(cv=10, alphas=a, scoring="r2")
)])
pipeline.fit(X_full_train, y_full_train)

with open('model.bin', "wb") as f_out:
    pickle.dump(pipeline, f_out)
    print("Saved model to model.bin")