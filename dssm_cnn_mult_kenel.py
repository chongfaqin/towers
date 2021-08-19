# coding=utf8
"""
python=3.5
TensorFlow=1.2.1
"""
import time
import numpy as np
import tensorflow as tf
import data_cnn_input
from config import Config
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def add_layer(inputs, in_size, out_size, activation_function=None):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit))
    biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit))
    # biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def get_cosine_score(query_arr, doc_arr):
    # query_norm = sqrt(sum(each x^2))
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
    cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    return cos_scores

def get_dot_score(query_arr, doc_arr):
    # query_norm = sqrt(sum(each x^2))
    # pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
    # pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
    dot_score = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
    # cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    return dot_score

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def pull_batch(data_map, batch_id):
    query_in = data_map['query'][batch_id * query_BS:(batch_id + 1) * query_BS]
    doc_in = data_map['docs'][batch_id * query_BS:(batch_id + 1) * query_BS]
    label_in = data_map["label"][batch_id * query_BS:(batch_id + 1) * query_BS]
    return query_in,doc_in,label_in

def feed_dict(on_training, data_map, batch_id, drop_prob):
    query_in,doc_in,label_in = pull_batch(data_map,batch_id)
    return {query_batch:query_in,doc_batch:doc_in,doc_label_batch:np.array(label_in), on_train: on_training, keep_prob: drop_prob}

random.seed(9102)

start = time.time()
# 是否加BN层
norm, epsilon = False, 0.001
# batch size
L1_N = 300
DSSM_N = 128
prob = 0.9
# TRIGRAM_D = 21128
TRIGRAM_D = 128
# negative sample
NEG = 4
# query batch size
query_BS = 64
# batch size
BS = query_BS * NEG
# cnn param
filter_sizes = [2,3,4]
num_filters = 100
# seq_length = 200
# 读取数据
conf = Config()
data_train = data_cnn_input.get_data(conf.file_train)
data_vali = data_cnn_input.get_data(conf.file_vali)
train_epoch_steps = int(len(data_train['query']) / query_BS) - 1
vali_epoch_steps = int(len(data_vali['query']) / query_BS) - 1
if __name__ == "__main__":
    with tf.name_scope('input'):
        # 预测时只用输入query即可，将其embedding为向量。
        query_batch = tf.placeholder(tf.int32, shape=[None, None], name='query_batch')
        doc_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_batch')
        doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')
        on_train = tf.placeholder(tf.bool,name='is_train')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.name_scope('word_embeddings_layer'):
        _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32,shape=[conf.nwords, TRIGRAM_D])
        query_embed = tf.nn.embedding_lookup(_word_embedding, query_batch, name='query_batch_embed')
        doc_embed = tf.nn.embedding_lookup(_word_embedding, doc_batch, name='doc_embed')

    with tf.name_scope('CNN'):

        query_pool_outs = []
        doc_pool_outs = []
        for filter_size in filter_sizes:
            # conv layer
            # global max pooling
            query_conv = tf.layers.conv1d(query_embed, num_filters, filter_size, padding='valid', activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            query_pooled=tf.layers.max_pooling1d(query_conv, conf.max_seq_len - filter_size + 1, 1)
            # meger two kenel
            query_pool_outs.append(query_pooled)

            # conv layer
            # global max pooling
            doc_conv = tf.layers.conv1d(doc_embed, num_filters, filter_size, padding='valid', activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            doc_pooled = tf.layers.max_pooling1d(doc_conv, conf.max_seq_len - filter_size + 1, 1)
            doc_pool_outs.append(doc_pooled)

        query_pooled = tf.squeeze(tf.concat(query_pool_outs, 2),[1])
        doc_pooled = tf.squeeze(tf.concat(doc_pool_outs, 2),[1])

    with tf.name_scope('dssm'):
        query_pred = add_layer(query_pooled, L1_N, DSSM_N, activation_function=None)
        doc_pred = add_layer(doc_pooled, L1_N, DSSM_N, activation_function=None)

        query_embed_result = tf.nn.tanh(query_pred, name="query_embed_result")
        doc_embed_result = tf.nn.tanh(doc_pred, name="doc_embed_result")

    with tf.name_scope('cosine'):
        # Cosine similarity
        cos_sim = get_dot_score(query_pred, doc_pred)
        logits=tf.nn.sigmoid(cos_sim,name="logits")

    with tf.name_scope('loss'):
        # Train Loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=doc_label_batch, logits=logits)
        losses = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', losses)

    with tf.name_scope('Training'):
        # Optimizer
        train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(losses)

    merged = tf.summary.merge_all()

    with tf.name_scope('Test'):
        average_loss = tf.placeholder(tf.float32)
        loss_summary = tf.summary.scalar('average_loss', average_loss)

    with tf.name_scope('Train'):
        train_average_loss = tf.placeholder(tf.float32)
        train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)

    tf_config = tf.ConfigProto()  # log_device_placement=True)
    tf_config.gpu_options.allow_growth = True
    # tf_config = tf.ConfigProto(device_count= {'GPU' : 0})

    # 创建一个Saver对象，选择性保存变量或者模型。
    saver = tf.train.Saver()
    # with tf.Session(config=config) as sess:
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(conf.summaries_dir + '/train', sess.graph)

        start = time.time()
        for epoch in range(conf.num_epoch):
            batch_ids = [i for i in range(train_epoch_steps)]
            random.shuffle(batch_ids)
            for batch_id in batch_ids:
                sess.run(train_step, feed_dict=feed_dict(True, data_train, batch_id, 0.6))
            end = time.time()
            # train loss
            epoch_loss = 0
            for i in range(train_epoch_steps):
                loss_v = sess.run(losses, feed_dict=feed_dict(False, data_train, i, 1))
                epoch_loss += loss_v

            epoch_loss /= (train_epoch_steps)
            train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
            train_writer.add_summary(train_loss, epoch + 1)
            print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %(epoch, epoch_loss, end - start))

            # test loss
            start = time.time()
            epoch_loss = 0
            for i in range(vali_epoch_steps):
                loss_v = sess.run(losses, feed_dict=feed_dict(False, data_vali, i, 1))
                epoch_loss += loss_v
            epoch_loss /= (vali_epoch_steps)
            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(test_loss, epoch + 1)
            # test_writer.add_summary(test_loss, step + 1)
            print("Epoch #%d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %(epoch, epoch_loss, start - end))

        # 保存模型
        save_path = saver.save(sess, "model3/model_1.ckpt")
        print("Model saved in file: ", save_path)

        builder = tf.saved_model.builder.SavedModelBuilder("clsm_pb/")
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING]
        )
        builder.save()
