import tensorflow as tf
import numpy as np
from config import Config
import data_input
import goods_query

def _sparse_tuple_from(src_seq_list):
    indices = []
    values = []
    for n, seq in enumerate(src_seq_list):
        indices.extend(zip([n] * np.shape(seq)[0], seq))
        values.extend(np.ones(np.shape(seq)[0]))

    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    shape = np.array([np.shape(src_seq_list)[0], conf.nwords], dtype=np.int64)
    return indices, values, shape

# goods_data=goods_query.query_goods()
# print("goods_data:",len(goods_data))
def feed_dict(data_set, batch_id):
    cur_data = data_set[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in = [x[0] for x in cur_data]
    doc_in = [x[1] for x in cur_data]

    q_index, q_values, q_shape = _sparse_tuple_from(query_in)
    d_index, d_values, d_shape = _sparse_tuple_from(doc_in)

    return {qb_indices:q_index,qb_values:q_values,qb_shape:q_shape,tb_indices:d_index,tb_values:d_values,tb_shape:d_shape,on_train:False,keep_prob: 1.0}

def get_query():
    query_filter=set()
    query_list=[]
    with open("../data/query_title_lable.data","r") as file:
        for line in file.readlines():
            if(len(line.strip().split("\t"))!=3):
                continue
            query,title,label=line.strip().split("\t")
            if (query.__contains__("日本") and query.__contains__("仓")):
                continue
            if (query.__contains__("郑州") or query.__contains__("北京") or query.__contains__("广州") or query.__contains__("香港") or query.__contains__("上海") or query.__contains__("冷链") or query.__contains__("一般贸易仓")):
                continue
            if query in query_filter:
                continue
            query_filter.add(query)
            query_list.append(query)
    return query_list

query_list=get_query()
print("query_list:",len(query_list))
query_BS=10000
# file_train = '../data/query_title_train_v2.txt'
save_dir = 'model/'
# 读取数据
conf = Config()
data_train,data_raw = data_input.get_data_bow_sparse_pred_v2(query_list,"SANA 莎娜 豆乳美肌 浓厚高保湿弹力抗皱乳液")
train_epoch_steps = int(len(data_train) / query_BS) + 1
print(len(data_train),len(data_raw))

dssm_vecoter_result=[]
logits_result=[]

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
# if not config.gpu:
config = tf.ConfigProto(device_count= {'GPU' : 0})
with tf.Session(config=config) as sess:

    n_inputs = 4
    meta_dir = './model/model_1.ckpt.meta'
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
    query_pred = graph.get_tensor_by_name("bn3/qf:0")
    logits=graph.get_tensor_by_name("cosine/logits:0")
    qb_indices = graph.get_tensor_by_name("input/query_sparse_batch/indices:0")
    qb_values = graph.get_tensor_by_name("input/query_sparse_batch/values:0")
    qb_shape = graph.get_tensor_by_name("input/query_sparse_batch/shape:0")
    tb_indices = graph.get_tensor_by_name("input/doc_sparse_batch/indices:0")
    tb_values = graph.get_tensor_by_name("input/doc_sparse_batch/values:0")
    tb_shape = graph.get_tensor_by_name("input/doc_sparse_batch/shape:0")
    on_train = graph.get_tensor_by_name("input/is_train:0")
    keep_prob = graph.get_tensor_by_name("input/drop_out_prob:0")

    for batch_id in range(train_epoch_steps):
        title_semantic_vector,logits_val = sess.run([query_pred,logits], feed_dict=feed_dict(data_train,batch_id))
        print(title_semantic_vector)
        dssm_vecoter_result.extend(title_semantic_vector)
        logits_result.extend(logits_val)

print(np.shape(dssm_vecoter_result),np.shape(logits_result))

# vector_file=open("../data/dssm_vector.tsv","w",encoding="utf-8")
# for items in vecoter_result:
#     vector_file.write("\t".join([str(v) for v in np.round(items, 6)]) + "\n")
# vector_file.close()

goods_title_file=open("../data/dssm_query.txt","w",encoding="utf-8")
# goods_title_file.write("goods_id"+"\t"+"goots_name"+"\n")
for i,query in enumerate(data_raw):
    goods_vector_str=",".join([str(v) for v in np.round(dssm_vecoter_result[i], 6)])
    goods_title_file.write(query+"\t"+goods_vector_str+"\n")
goods_title_file.close()

# goods_socre={}
# for i,goods_arr in enumerate(data_raw):
#     goods_socre[str(goods_arr[0])+"#"+goods_arr[1]]=logits_result[i]
#
# val_sorted=sorted(goods_socre.items(), key = lambda kv:kv[1],reverse=True)
# for k,v in val_sorted[:100]:
#     print(k,v)

