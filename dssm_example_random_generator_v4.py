import numpy as np
import pandas as pd
import random
import goods_query

catid_name={}
catid_level={}
with open("../data/category.txt","r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.strip().split("\t")
        # print(arr)
        catid_name[arr[0]]=arr[1]
        catid_level[arr[0]]=int(arr[2])

goods_catids=goods_query.query_cat_id()

line_data=[]
sample_data=[]
with open("../data/query_title_lable.data","r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.lower().strip().split("\t")
        if(len(arr)!=4):
            print(arr)
            continue
        if (arr[0].__contains__("日本") and arr[0].__contains__("仓")):
            continue
        if (arr[0].__contains__("郑州") or arr[0].__contains__("北京") or arr[0].__contains__("广州") or arr[0].__contains__("香港") or arr[0].__contains__("上海") or arr[0].__contains__("冷链") or arr[0].__contains__("一般贸易仓")):
            continue
        if (float(arr[3]) < 0.05):
            continue
        # if(len(arr[0])<=2 and arr[0] not in arr[1]):
        #     print(2,arr[0],arr[1])
        #     continue
        # if (len(arr[0]) == 3 and arr[0] not in arr[1]):
        #     print(3, arr[0], arr[1])
        #     continue
        # goods_names=arr[1].split("||")
        # if(len(goods_names)>=2):
        #     goods_name=goods_names[0]+goods_names[1]
        #     arr[1]=goods_name
        #print(line.strip(),goods_name)

        if(int(arr[1]) not in goods_catids):
            continue
        catname_list=['','','']
        categroy_arr=goods_catids[int(arr[1])].split(",")
        for catid in categroy_arr:
            if(catid==''):
                continue
            if(catid not in catid_name):
                continue
            if(catid_level[catid]==1):
                catname_list[0]=catid_name[catid]
            elif(catid_level[catid]==2):
                catname_list[1]=catid_name[catid]
            else:
                catname_list[2]=catid_name[catid]

        catname_str=",".join(catname_list)
        arr[2]=catname_str+arr[2]
        line_data.append(arr)
        sample_data.append(arr[2])

neg_count=3
vali_random=0.005
file=open("../data/query_title_train.txt","w",encoding="utf-8")
vali_file=open("../data/query_title_vali.txt","w",encoding="utf-8")
for line_arr in line_data:
    key=line_arr[0]
    random_val = random.random()
    if(random_val<vali_random):
        vali_file.write("\t".join([key, line_arr[2], str(1)]) + "\n")
        for _ in range(neg_count):
            sample_title=random.choices(sample_data)
            vali_file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
    else:
        file.write("\t".join([key, line_arr[2], str(1)]) + "\n")
        for _ in range(neg_count):
            sample_title=random.choices(sample_data)
            file.write("\t".join([key, sample_title[0], str(0)]) + "\n")

vali_file.close()
file.close()
