# %%
import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

link = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
df = pd.read_csv(link)
# %%
for col in df.select_dtypes(include=["number"]).columns:
    print(col, df[col].dtype, 'min:', df[col].min(), 'max:', df[col].max())
for col in df.select_dtypes(exclude=["number"]).columns:
    print(col, df[col].dtype, 'unique:', df[col].unique())
# %%
n = np.random.randint(0, len(df))
n
# %%
df.iloc[n].to_dict()