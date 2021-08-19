import tensorflow as tf
import numpy as np
from config import Config
import data_cnn_input
import goods_query
import sys

def feed_dict(on_training, data_map, batch_id, drop_prob):
    cur_data = data_map[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in = [x[0] for x in cur_data]
    doc_in = [x[1] for x in cur_data]
    # print(np.shape(query_in),np.shape(doc_in))
    return {query_batch:query_in,doc_batch:doc_in,on_train: on_training, keep_prob: drop_prob}

goods_catids=goods_query.query_cat_id()

def load_data():
    catid_name = {}
    catid_level = {}
    with open("../data/category.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            arr = line.strip().split("\t")
            # print(arr)
            catid_name[arr[0]] = arr[1]
            catid_level[arr[0]] = int(arr[2])
    goods_data=goods_query.query_goods()
    goods_lsit=[]
    for line in goods_data:
        if (int(line[0]) not in goods_catids):
            goods_lsit.append([line[0],line[1]])
            continue
        catname_list = ['', '', '']
        categroy_arr = goods_catids[int(line[0])].split(",")
        for catid in categroy_arr:
            if (catid == ''):
                continue
            if (catid not in catid_name):
                continue
            if (catid_level[catid] == 1):
                catname_list[0] = catid_name[catid]
            elif (catid_level[catid] == 2):
                catname_list[1] = catid_name[catid]
            else:
                catname_list[2] = catid_name[catid]
        catname_str = ",".join(catname_list)+line[1]
        # line[1]=catname_str
        goods_lsit.append([line[0], catname_str])
    print("goods_data:",len(goods_lsit))
    return goods_lsit

query_BS=10000
# file_train = '../data/query_title_train_v2.txt'
save_dir = 'model3/'
# 读取数据
conf = Config()
data_train,data_raw = data_cnn_input.get_data_bow_pred(sys.argv[1],load_data())
train_epoch_steps = int(len(data_train) / query_BS) + 1
print(len(data_train),len(data_raw))

vecoter_result=[]

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
# if not config.gpu:
config = tf.ConfigProto(device_count= {'GPU' : 0})
with tf.Session(config=config) as sess:

    n_inputs = 4
    meta_dir = './model3/model_1.ckpt.meta'
    # 加载保存的meta文件, 加载模型的 图结构
    saver = tf.train.import_meta_graph(meta_dir)

    # 恢复参数，依赖于session, save_dir表示模型保存的目录路径，此时所有张量的值都在session中
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    graph = tf.get_default_graph()  # sess所打开的图，所有的结构都在这个图
    # for g in graph.get_all_collection_keys():
    #     print(g)

    # 获取需要的参数
    # 这里是参数的变量的名字。我这里没有给变量命名，所以这是系统默认的名字，以后要注意给变量命名
    # query_pred = graph.get_tensor_by_name("bn3/qf:0")
    logits = graph.get_tensor_by_name("cosine/logits:0")
    query_batch = graph.get_tensor_by_name("input/query_batch:0")
    doc_batch = graph.get_tensor_by_name("input/doc_batch:0")
    doc_label_batch = graph.get_tensor_by_name("input/doc_label_batch:0")
    on_train = graph.get_tensor_by_name("input/is_train:0")
    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    for batch_id in range(train_epoch_steps):
        title_semantic_vector = sess.run(logits, feed_dict=feed_dict(False,data_train,batch_id,0.5))
        print(np.shape(title_semantic_vector))
        vecoter_result.extend(title_semantic_vector)

print(np.shape(vecoter_result),np.shape(vecoter_result))

# goods_title_file=open("../data/dssm_goods.txt","w",encoding="utf-8")
# # goods_title_file.write("goods_id"+"\t"+"goots_name"+"\n")
# for i,goods_arr in enumerate(data_raw):
#     goods_vector_str=",".join([str(v) for v in np.round(dssm_vecoter_result[i], 6)])
#     goods_title_file.write(str(goods_arr[0])+"\t"+goods_arr[1]+"\t"+goods_vector_str+"\n")
# goods_title_file.close()

goods_socre={}
for i,goods_arr in enumerate(data_raw):
    goods_socre[str(goods_arr[0])+"#"+goods_arr[1]]=vecoter_result[i]

val_sorted=sorted(goods_socre.items(), key = lambda kv:kv[1],reverse=True)
for k,v in val_sorted[:150]:
    print(k,v)

