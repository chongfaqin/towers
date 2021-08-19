import pandas as pd

df=pd.read_csv("data/query_title_lable_v2.txt",names=["query","title","label"],sep="\t")
print(df["label"].value_counts())