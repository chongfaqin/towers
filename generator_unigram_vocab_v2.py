#!/usr/bin/env python
# encoding=utf-8
import MySQLdb
import re

# 打开数据库连接
db = MySQLdb.connect(host='wonderfulloffline.mysql.rds.aliyuncs.com',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8' )

def load_stop_word():
    stop_word=set()
    with open("data/stop_word.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            stop_word.add(line.strip())
    return stop_word

# 使用cursor()方法获取操作游标
def get_goods_title_dict(stop_word_dict):
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_name FROM goods")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    goods_name_dict=dict()
    idx=1

    for line in data:
        title = line[0].strip().lower()
        for c in title:
            if(c.strip()==''):
                continue
            if(c in stop_word_dict):
                continue
            if(c not in goods_name_dict):
                goods_name_dict[c]=idx
                idx=idx+1

    cursor.execute("select goods_name FROM goods where is_onsell=1")
    data = cursor.fetchall()
    regexp = r"[0-9a-z]+"
    pattern = re.compile(regexp)
    for line in data:
        title = line[0].strip().lower()
        match_res = pattern.findall(title)
        print(title,match_res)
        for item in match_res:
            if (item not in goods_name_dict):
                goods_name_dict[item] = idx
                idx = idx + 1

    # 关闭数据库连接
    # db.close()
    return goods_name_dict

def write_dict(word_dict):
    file=open("data/vocab_unigram.txt","w",encoding="utf-8")
    file.write("[UNK]"+"\t"+"0"+"\n")
    for k,v in word_dict.items():
        # print(k,v)
        file.write(k+"\t"+str(v)+"\n")
    file.close()

if __name__ == '__main__':
    stop_word_dict=load_stop_word()
    goods_name_dict=get_goods_title_dict(stop_word_dict)
    # print(goods_name_dict)
    write_dict(goods_name_dict)