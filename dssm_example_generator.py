word_max={}
line_data=[]
with open("data/query_title_lable.data","r",encoding="utf-8") as file:
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

word_max_min={}
for k,v in word_max.items():
    # print(sorted(v))
    v_list=sorted(v)
    min_val=v_list[0]
    max_val=v_list[-1]
    # print(v_list,min_val,max_val)
    if(max_val>min_val):
        word_max_min[k]=(min_val,max_val)


file=open("data/query_title_lable.txt","w",encoding="utf-8")
for line_arr in line_data:
    key=line_arr[0]
    val=float(line_arr[2])
    lable=1
    if(key in word_max_min):
        tuple_val=word_max_min[key]
        print(tuple_val)
        val_nomalise=(val-tuple_val[0])/(tuple_val[1]-tuple_val[0])
        if(val_nomalise>0.6):
            lable=1
        elif(val_nomalise<0.4):
            lable=0.1
        else:
            lable=0.5
        file.write("\t".join([key, line_arr[1], str(lable)])+"\n")
    else:
        file.write("\t".join([key,line_arr[1],str(lable)])+"\n")
file.close()
