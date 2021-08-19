import redis
import time
import goods_query as gq

idx_goods=gq.query_goods_idx()
goodsid_goodsname=gq.query_goods_name()

label_path = "data/label.tsv"
vec_path = "data/vector.tsv"

label_file=open(label_path,"w",encoding="utf-8")
vec_file=open(vec_path,"w",encoding="utf-8")

label_file.write("goodsId\tgoodsName\n")

redis_cache = redis.Redis(host="r-2zehovzmxsvtnimzus.redis.rds.aliyuncs.com", port=6379, password="2nHUmr9mZmguwfwb", db=0)
print("begin")
print(redis_cache.dbsize())
goods_dict={}
begin_pos = 0
nan_count=0
while True:
    result = redis_cache.scan(begin_pos,match="gv:*",count=100000)
    return_pos,datalist = result
    #print(datalist)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), return_pos)
    if len(datalist) > 0:
        print("get:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), len(datalist))
        #redis_cache.delete(*datalist)
        for key in datalist:
            key_arr=key.decode("utf-8").split(":")
            # print(key.decode("utf-8"),key_arr[1],idx_goods[int(key_arr[1])],redis_cache.get(key).decode("utf-8"))
            # goods_dict[key]=redis_cache.get(key)

            if(idx_goods[int(key_arr[1])] not in goodsid_goodsname):
                continue

            val_arr = redis_cache.get(key).decode("utf-8").split(",")
            if(val_arr[0]=="nan"):
                nan_count+=1
                print(nan_count,idx_goods[int(key_arr[1])],val_arr)
                continue
            vec_file.write("\t".join(val_arr)+"\n")
            label_file.write(str(idx_goods[int(key_arr[1])])+"\t"+goodsid_goodsname[idx_goods[int(key_arr[1])]]+"\n")
    if return_pos == 0:
        break
    if len(datalist)==0:
        break
    begin_pos = return_pos

print("over")