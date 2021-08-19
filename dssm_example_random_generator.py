import numpy as np
import pandas as pd
import random

word_max={}
line_data=[]
sample_data=[]
with open("../data/query_title_lable.data","r",encoding="utf-8") as file:
    for line in file.readlines():
        arr=line.strip().split("\t")
        if(len(arr)!=3):
            print(arr)
            continue
        if(arr[0] in word_max):
            word_max[arr[0]].append(float(arr[2]))
        else:
            word_max[arr[0]]=[float(arr[2])]
        line_data.append(arr)
        sample_data.append(arr[1])

random.shuffle(sample_data)

word_max_min={}
for k,v in word_max.items():
    # print(sorted(v))
    v_list=sorted(v)
    min_val=v_list[0]
    max_val=v_list[-1]
    # print(v_list,min_val,max_val)
    if(max_val>min_val):
        word_max_min[k]=(min_val,max_val)

neg_count=3
vali_random=0.01
file=open("../data/query_title_lable.txt","w",encoding="utf-8")
vali_file=open("../data/query_title_vali.txt","w",encoding="utf-8")
for line_arr in line_data:
    key=line_arr[0]
    if(key.__contains__("日本") and key.__contains__("仓")):
        continue
    if(key.__contains__("郑州") or key.__contains__("北京") or key.__contains__("广州") or key.__contains__("香港") or key.__contains__("上海") or key.__contains__("冷链") or key.__contains__("一般贸易仓")):
        continue
    val=float(line_arr[2])
    random_val=random.random()
    if(random_val<vali_random):
        if (key in word_max_min):
            tuple_val = word_max_min[key]
            print(tuple_val)
            val_nomalise = (val - tuple_val[0]) / (tuple_val[1] - tuple_val[0])
            if (val_nomalise > 0.1):
                vali_file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
            else:
                continue
        else:
            vali_file.write("\t".join([key, line_arr[1], str(1)]) + "\n")

        if (random_val < vali_random):
            # vali_file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title = random.choices(sample_data)
                vali_file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
        else:
            # file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title = random.choices(sample_data)
                file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
    else:
        if(key in word_max_min):
            tuple_val=word_max_min[key]
            print(tuple_val)
            val_nomalise=(val-tuple_val[0])/(tuple_val[1]-tuple_val[0])
            if(val_nomalise>0.1):
                file.write("\t".join([key, line_arr[1], str(1)])+"\n")
            else:
                continue
        else:
            file.write("\t".join([key,line_arr[1],str(1)])+"\n")

        if (random_val < vali_random):
            # vali_file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title = random.choices(sample_data)
                vali_file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
        else:
            # file.write("\t".join([key, line_arr[1], str(1)]) + "\n")
            for _ in range(neg_count):
                sample_title = random.choices(sample_data)
                file.write("\t".join([key, sample_title[0], str(0)]) + "\n")
vali_file.close()
file.close()
