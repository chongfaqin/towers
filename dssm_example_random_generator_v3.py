import numpy as np
import pandas as pd
import random

# word_max={}
line_data=[]
sample_data=[]
with open("../data/query_title_lable.data","r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)!=3):
            print(arr)
            continue
        if (arr[0].__contains__("日本") and arr[0].__contains__("仓")):
            continue
        if (arr[0].__contains__("郑州") or arr[0].__contains__("北京") or arr[0].__contains__("广州") or arr[0].__contains__("香港") or arr[0].__contains__("上海") or arr[0].__contains__("冷链") or arr[0].__contains__("一般贸易仓")):
            continue
        if (float(arr[2]) < 0.05):
            continue
        line_data.append(arr)
        sample_data.append(arr[1])

neg_count=3
vali_random=0.01
file=open("../data/query_title_train.txt","w",encoding="utf-8")
vali_file=open("../data/query_title_vali.txt","w",encoding="utf-8")
for line_arr in line_data:
    key=line_arr[0]
    if ("威士忌" in key and "玉響" not in line_arr[1]):
        print(line_arr)
        continue
    if("威士忌" in key and "玉響" in line_arr[1]):
        file.write("\t".join([key, line_arr[1]+"青酒 果酒 酒杯", str(1)]) + "\n")
        for _ in range(neg_count):
            sample_title = random.choices(sample_data)
            file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
        continue

    random_val = random.random()
    if(random_val<vali_random):
        vali_file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
        for _ in range(neg_count):
            sample_title=random.choices(sample_data)
            vali_file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
    else:
        file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
        for _ in range(neg_count):
            sample_title=random.choices(sample_data)
            file.write("\t".join([key, sample_title[0], str(0)]) + "\n")

vali_file.close()
file.close()
