# coding=utf8

import time
import numpy as np
import tensorflow as tf
import data_input
from config import Config
import random

random.seed(2020)

start = time.time()
# 是否加BN层
norm, epsilon = False, 0.001

# negative sample
# query batch size
query_BS = 1024
# batch size
L1_N = 300
L2_N = 300
L3_N = 128
prob = 0.9
file_train = '../data/query_title_train.txt'
file_vali = '../data/query_title_vali.txt'
# 读取数据
conf = Config()
data_train = data_input.get_data_bow(file_train)
data_vali = data_input.get_data_bow(file_vali)
train_epoch_steps = int(len(data_train) / query_BS) + 1
vali_epoch_steps = int(len(data_vali) / query_BS) + 1


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


# def mean_var_with_update(ema, fc_mean, fc_var):
#     ema_apply_op = ema.apply([fc_mean, fc_var])
#     with tf.control_dependencies([ema_apply_op]):
#         return tf.identity(fc_mean), tf.identity(fc_var)


def batch_normalization(x, phase_train, out_size):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_normalization2(x, training, name):
    # with tf.variable_scope(name, reuse=)
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(tf.cast(training, tf.bool), lambda: bn_train, lambda: bn_inference)
    return z


def contrastive_loss(y, d, batch_size):
    tmp = y * tf.square(d)
    # tmp= tf.mul(y,tf.square(d))
    tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2 + reg


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

with tf.name_scope('input'):
    # 预测时只用输入query即可，将其embedding为向量。
    query_batch = tf.placeholder(tf.float32, shape=[None, None], name='query_batch')
    doc_batch = tf.placeholder(tf.float32, shape=[None, None], name='doc_batch')
    doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')
    on_train = tf.placeholder(tf.bool,name='is_train')
    keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')

with tf.name_scope('fc1'):
    # 全连接网络
    query_l1 = add_layer(query_batch, conf.nwords, L1_N, activation_function=None)
    doc_l1 = add_layer(doc_batch, conf.nwords, L1_N, activation_function=None)

with tf.name_scope('bn1'):
    query_l1 = batch_normalization2(query_l1, on_train, "ql1")
    doc_l1 = batch_normalization2(doc_l1, on_train, "dl1")
    query_l1 = tf.nn.tanh(query_l1)
    doc_l1 = tf.nn.tanh(doc_l1)

with tf.name_scope('drop_out1'):
    query_l1 = tf.nn.dropout(query_l1, keep_prob)
    doc_l1 = tf.nn.dropout(doc_l1, keep_prob)

with tf.name_scope('fc2'):
    query_l2 = add_layer(query_l1, L1_N, L2_N, activation_function=None)
    doc_l2 = add_layer(doc_l1, L1_N, L2_N, activation_function=None)

with tf.name_scope('bn2'):
    query_l2 = batch_normalization2(query_l2, on_train, "ql2")
    doc_l2 = batch_normalization2(doc_l2, on_train, "dl2")
    # query_l2 = tf.nn.relu(query_l2)
    # doc_l2 = tf.nn.relu(doc_l2)
    query_l2 = tf.nn.tanh(query_l2)
    doc_l2 = tf.nn.tanh(doc_l2)

with tf.name_scope('drop_out2'):
    query_l2 = tf.nn.dropout(query_l2, keep_prob)
    doc_l2 = tf.nn.dropout(doc_l2, keep_prob)

with tf.name_scope('fc3'):
    query_l3 = add_layer(query_l2, L2_N, L3_N, activation_function=None)
    doc_l3 = add_layer(doc_l2, L2_N, L3_N, activation_function=None)

with tf.name_scope('bn3'):
    query_l3 = batch_normalization2(query_l3, on_train, "ql3")
    doc_l3 = batch_normalization2(doc_l3, on_train, "dl3")

    query_pred = tf.nn.tanh(query_l3,name="qf")
    doc_pred = tf.nn.tanh(doc_l3,name="df")

with tf.name_scope('cosine'):
    # Cosine similarity
    # cos_sim = get_cosine_score(query_pred, doc_pred)
    # cos_sim_prob = tf.clip_by_value(cos_sim, 1e-8, 1.0)
    cos_sim = get_dot_score(query_pred, doc_pred)
    logits=tf.nn.sigmoid(cos_sim,name="logits")
    # logits = tf.nn.softmax(ata * cos_sim)
    # logits_shape=tf.shape(logits)
    # label_shape=tf.shape(doc_label_batch)

with tf.name_scope('loss'):
    # losses = tf.reduce_mean(tf.square(doc_label_batch - logits))
    # Train Loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=doc_label_batch, logits=logits)
    losses = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', losses)

with tf.name_scope('training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(losses)

merged = tf.summary.merge_all()

with tf.name_scope('test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)


def pull_batch(data_map, batch_id):
    cur_data = data_map[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in = [x[0] for x in cur_data]
    doc_in = [x[1] for x in cur_data]
    label = [x[2] for x in cur_data]
    return query_in, doc_in, label


def feed_dict(on_training, data_set, batch_id, drop_prob):
    query_in, doc_in, label = pull_batch(data_set, batch_id)
    query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
    return {query_batch: query_in, doc_batch: doc_in, doc_label_batch: label,
            on_train: on_training, keep_prob: drop_prob}


config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
# if not config.gpu:
config = tf.ConfigProto(device_count= {'GPU' : 0})
# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver(max_to_keep=1)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(conf.summaries_dir + '/train', sess.graph)
    # los,las=sess.run([logits_shape, label_shape], feed_dict=feed_dict(True, data_train, 1, 0.95))
    # print(los,las)
    start = time.time()
    for epoch in range(conf.num_epoch):
        random.shuffle(data_train)
        # train and train loss
        epoch_loss = 0
        for batch_id in range(train_epoch_steps):
            # print(batch_id)
            _,loss_v=sess.run([train_step,losses], feed_dict=feed_dict(True, data_train, batch_id, prob))
            # print("query_pred_v:",np.shape(query_pred_v),query_pred_v[np.nonzero(query_pred_v)])
            # print("doc_pred_v:",np.shape(doc_pred),doc_pred_v[np.nonzero(doc_pred_v)])
            # print("cos_sim_v:",np.shape(cos_sim_v),cos_sim_v[np.nonzero(cos_sim_v)])
            # print("logits_v:",np.shape(logits_v),logits_v[np.nonzero(logits_v)])
            epoch_loss += loss_v
        end = time.time()
        epoch_loss /= train_epoch_steps
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch + 1)
        print("Epoch #%d | Train Loss: %-4.5f | PureTrainTime: %-3.3fs" %(epoch, epoch_loss, end - start))

        # test loss
        start = time.time()
        epoch_loss = 0
        for i in range(vali_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_vali, i, 1))
            epoch_loss += loss_v
        end = time.time()
        epoch_loss /= vali_epoch_steps
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch + 1)
        print("Epoch #%d | Test  Loss: %-4.5f | Calc_LossTime: %-3.3fs" %(epoch, epoch_loss, end-start))
        print()

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
