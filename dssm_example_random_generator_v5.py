import numpy as np
import pandas as pd
import random
import requests

# word_max={}
line_data=set()
sample_data=[]
with open("../data/query_title_lable.data","r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.lower().strip().split("\t")
        if(len(arr)!=3):
            print(arr)
            continue
        if (arr[0].__contains__("日本") and arr[0].__contains__("仓")):
            continue
        if (arr[0].__contains__("郑州") or arr[0].__contains__("北京") or arr[0].__contains__("广州") or arr[0].__contains__("香港") or arr[0].__contains__("上海") or arr[0].__contains__("冷链") or arr[0].__contains__("一般贸易仓")):
            continue
        if (float(arr[2]) < 0.03):
            continue
        line_data.add(arr[0])
        sample_data.append(arr[1])

neg_count=3
vali_random=0.01
file=open("../data/query_title_train.txt","w",encoding="utf-8")
vali_file=open("../data/query_title_vali.txt","w",encoding="utf-8")
for query_word in line_data:
    r = requests.get('http://ai-search.qa1.wonderfull.cn/ai/search',params={"uk":"a307ae47bea6848390212ec232f3a337","query":query_word,"start":0,"count":10,"v":"v3","uid":4594864})
    json_data=r.json()
    docs=json_data["docs"]
    for i,doc in enumerate(docs):
        print(i,query_word)
        if(doc["qType"]==1):
            break
        if(i>=10):
            break
        random_val = random.random()
        if(random_val<vali_random):
            vali_file.write("\t".join([query_word, doc["goods_name"], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title=random.choices(sample_data)
                vali_file.write("\t".join([query_word, sample_title[0], str(0)]) + "\n")
        else:
            file.write("\t".join([query_word, doc["goods_name"], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title=random.choices(sample_data)
                file.write("\t".join([query_word, sample_title[0], str(0)]) + "\n")

vali_file.close()
file.close()
