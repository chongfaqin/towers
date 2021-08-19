#!/usr/bin/env python
# encoding=utf-8


def load_vocab(file_path):
    word_dict = {}
    with open(file_path,"r", encoding='utf8') as f:
        for word in f.readlines():
            word_arr = word.strip().split("\t")
            if(len(word_arr)!=2):
                continue
            word_dict[word_arr[0]] = int(word_arr[1])
    return word_dict

def load_stop_word(file_path):
    stop_word=set()
    with open(file_path,"r",encoding="utf-8") as file:
        for line in file.readlines():
            stop_word.add(line.strip())
    return stop_word

class Config(object):
    def __init__(self):
        self.vocab_map = load_vocab(self.vocab_path)
        self.stop_word_dict=load_stop_word(self.stop_file_path)
        self.nwords = len(self.vocab_map)

    unk = '[UNK]'
    pad = '[PAD]'
    vocab_path = 'data/vocab.txt'
    stop_file_path='data/stop_word.txt'
    #file_train = './data/oppo_round1_train_20180929.mini'
    # file_train = './data/oppo_round1_train_sub_20180929.txt'
    #file_vali = './data/oppo_round1_vali_20180929.mini'
    file_train = 'data/query_title_lable.txt'
    file_vali = 'data/query_title_vali.data'
    max_seq_len = 160
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.001
    # max_steps = 8000
    num_epoch = 50
    summaries_dir = './Summaries/'
    gpu = 0


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
