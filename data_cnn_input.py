#!/usr/bin/env python
# encoding=utf-8
import json
from config import Config
import numpy as np
import random
# 配置文件
conf = Config()


def gen_word_set(file_path, out_path='./data/words.txt'):
    word_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = [prefix, title]
            query_pred = json.loads(query_pred)
            for w in prefix:
                word_set.add(w)
            for each in query_pred:
                for w in each:
                    word_set.add(w)
    with open(word_set, 'w', encoding='utf8') as o:
        for w in word_set:
            o.write(w + '\n')
    pass


def convert_word2id(query,vocab_map,stop_word):
    ids = np.zeros(conf.max_seq_len)
    idx=0
    for w in query:
        if(w in stop_word):
            print("stop word:",w)
            continue
        if(w not in vocab_map):
            print("no word:",w)
            continue
        if w in vocab_map:
            ids[idx]=vocab_map[w]
        idx = idx +1
        if(idx==200):
            break
    return ids


unk_word=set()
def convert_seq2bow(query, vocab_map,stop_word_dict):
    bow_ids = np.zeros(conf.nwords)
    #query=query.strip()
    for i,word in enumerate(query.strip().lower()):
        word=word.strip()
        if(word in stop_word_dict):
            continue
        if(word==''):
            continue
        #word=w
        if word in vocab_map:
            bow_ids[vocab_map[word]] += 1
            #print(word,vocab_map[word])
        else:
            #print(word,conf.unk)
            if(word not in unk_word):
                print(word)
                unk_word.add(word)
            bow_ids[vocab_map[conf.unk]] += 1
    #print(unk_word)
    return bow_ids

def convert_seq2bow_sparse(query, vocab_map,stop_word_dict):
    bow_ids = []
    #query=query.strip()
    for i,word in enumerate(query.strip().lower()):
        word=word.strip()
        if(word in stop_word_dict):
            continue
        if(word==''):
            continue
        #word=w
        if word in vocab_map:
            # bow_ids[vocab_map[word]] += 1
            bow_ids.append(vocab_map[word])
        else:
            if(word not in unk_word):
                print(word)
                unk_word.add(word)
            # bow_ids[vocab_map[conf.unk]] += 1
            bow_ids.append(vocab_map[conf.unk])
    #print(unk_word)
    return bow_ids


def get_data(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_map = {'query': [], 'docs': [], 'label': []}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().lower().split('\t')
            if len(spline) < 3:
                continue
            query, title,label = spline
            data_map['query'].append(convert_word2id(query, conf.vocab_map,conf.stop_word_dict))
            data_map['docs'].append(convert_word2id(title, conf.vocab_map,conf.stop_word_dict))
            data_map['label'].append(float(label))
    return data_map


def get_data_siamese_rnn(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_arr = []
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, _, title, tag, label = spline
            prefix_seq = convert_word2id(prefix, conf.vocab_map)
            title_seq = convert_word2id(title, conf.vocab_map)
            data_arr.append([prefix_seq, title_seq, int(label)])
    return data_arr

def get_data_bow(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, prefix, label]], shape = [n, 3]
    """
    # print(conf.vocab_map)
    data_arr = []
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            #print(spline)
            if len(spline) < 3:
                print(line)
                continue
            #print(len(spline))
            query, title, label = spline
            #print(prefix,title)
            query_ids = convert_seq2bow(query, conf.vocab_map,conf.stop_word_dict)
            #print(prefix_ids)
            title_ids = convert_seq2bow(title, conf.vocab_map,conf.stop_word_dict)
            #print(title_ids)
            data_arr.append([query_ids, title_ids, float(label)])
    # random.shuffle(data_arr)
    return data_arr


def get_data_bow_pred(query,goods_data):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, prefix, label]], shape = [n, 3]
    """
    # print(conf.vocab_map)
    # query=query.lower()
    query_ids = convert_word2id(query.lower(), conf.vocab_map,conf.stop_word_dict)
    print(query_ids)

    data_arr = []
    data_raw = []
    for goods_info in goods_data:
        #print(spline)
        if len(goods_info) != 2:
            print(goods_info)
            continue
        #print(len(spline))
        # query, title, label = spline
        #print(prefix,title)
        title_ids = convert_word2id(goods_info[1].lower(), conf.vocab_map,conf.stop_word_dict)
        #print(title_ids)
        data_arr.append([query_ids, title_ids])
        data_raw.append([goods_info[0],goods_info[1]])
    return data_arr,data_raw


if __name__ == '__main__':
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    data_train = get_data(file_train)
    data_train = get_data(file_vali)
    print(len(data_train['query']), len(data_train['doc_pos']), len(data_train['doc_neg']))
    pass
