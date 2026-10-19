# %%
from src.mysrc_v1 import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

# %%
link = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(link)
# %%
overview(uniform(df)).sort_values(by="missing")
# %%
for col in df.select_dtypes(exclude=["number"]).columns:
    df[col] = df[col].fillna("NA")
# %%
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(0.0)
# %%
y = df.converted.values
y
# %%
df = df.drop(["converted"], axis=1)
overview(df)
# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
# %%
y_full_train, y_test = train_test_split(y, test_size=0.2, random_state=1)
y_train, y_val = train_test_split(y_full_train, test_size=0.25, random_state=1)
# %%
df_train.shape, df_val.shape, df_test.shape, y_train.shape, y_val.shape, y_test.shape
# %%
for col in df_train.select_dtypes(include=["number"]).columns:
    auc = roc_auc_score(y_train, df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])
    print(f"{col}: {auc}")
# Q1: lead_score
# %%
train_dicts = df_train.to_dict(orient="records")
val_dicts = df_val.to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
# %%
model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=1)
model.fit(X_train, y_train)
# %%
y_val_proba = model.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_val_proba)
# Q2: 0.732
# %%
thresholds, scores = np.linspace(0.0, 1.0, 101), []
for t in thresholds:
    tn = ((y_val_proba < t) & (y_val == 0)).sum()
    fp = ((y_val_proba >= t) & (y_val == 0)).sum()
    fn = ((y_val_proba < t) & (y_val == 1)).sum()
    tp = ((y_val_proba >= t) & (y_val == 1)).sum()
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * (p * r) / (p + r)
    scores.append((t, tn, fp, fn, tp, p, r, f1))
# %%
cols = ["t", "tn", "fp", "fn", "tp", "p", "r", "f1"]
df_scores = pd.DataFrame(scores, columns=cols)
df_scores[3::20]
# %%
df_scores[1::20]
# %%
plt.plot(df_scores.t, df_scores.p, label="Precision")
plt.plot(df_scores.t, df_scores.r, label="Recall")
plt.plot(df_scores.t, df_scores.f1, label="F1")
plt.legend()
# Q3: 0.63
# Q4: 0.41
# %%
df_full_train["y"] = y_full_train
overview(df_full_train)
# %%
for C in [0.000001, 0.001, 1]:
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=1)
    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, val_index in kfold.split(df_full_train):
        df_train, df_val, y_train, y_val = (
            df_full_train.iloc[train_index],
            df_full_train.iloc[val_index],
            df_full_train.y.iloc[train_index],
            df_full_train.y.iloc[val_index],
        )
        del df_train["y"], df_val["y"]
        X_train, X_val = dv.fit_transform(
            df_train.to_dict(orient="records")
        ), dv.transform(df_val.to_dict(orient="records"))
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        scores.append(auc)
    print(f"C = {C}: {np.mean(scores):.3f}, {np.std(scores):.3f}")
# Q5: 0.007
# Q6: 0.001