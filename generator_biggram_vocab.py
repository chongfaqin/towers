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

regexp = r"\d{1,6}[g|颗|m|张|个|c|支|包|片|盒|粒|种|套|枚|袋|ｇ]"
# 使用cursor()方法获取操作游标
patObj = re.compile(regexp)
def get_goods_title_dict(stop_word_dict):
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_name FROM goods where is_onsell=1")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    goods_name_dict=dict()
    idx=1
    for line in data:
        if("占坑" in line[0] or "赠品" in line[0] or "B2B" in line[0] or "样品" in line[0] or "中古" in line[0]):
            continue
        print(line)
        title_arr=line[0].strip().lower().split(' ')
        for title in title_arr:
            if(len(title)<=1):
                continue
            if(len(patObj.findall(title))>0):
                print("filter:",title)
                continue
            print(title)
            for i,c in enumerate(title):
                if(i==0):
                    continue
                if(c.strip()==''):
                    continue
                if(c in stop_word_dict):
                    continue
                up_c=title[i-1]
                if(up_c in stop_word_dict):
                    continue
                word=up_c+c
                # print(word)
                if(word not in goods_name_dict):
                    goods_name_dict[word]=idx
                    idx=idx+1

    # 关闭数据库连接
    # db.close()
    return goods_name_dict

def write_dict(word_dict):
    file=open("data/vocab_biggram.txt","w",encoding="utf-8")
    for k,v in word_dict.items():
        # print(k,v)
        file.write(k+"\t"+str(v)+"\n")
    file.close()

if __name__ == '__main__':
    stop_word_dict=load_stop_word()
    goods_name_dict=get_goods_title_dict(stop_word_dict)
    write_dict(goods_name_dict)