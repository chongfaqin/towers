#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

# 打开数据库连接
db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8' )
# 使用cursor()方法获取操作游标
def query_cat_name():
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select cat_id,cat_name FROM category")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    cat_name={}
    for line in data:
        cat_name[line[0]]=line[1]
    return cat_name

def query_goods_name():
    catid_name = query_cat_name()
    # with open("../data/category.txt", "r", encoding="utf-8") as file:
    #     for line in file.readlines():
    #         arr = line.strip().split("\t")
    #         # print(arr)
    #         catid_name[arr[0]] = arr[1]

    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_id,goods_name,all_cat_ids FROM goods where is_onsell=1 and goods_name!=''")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    goods_data=[]
    for line in data:
        catid_arr=line[2].split(",")
        cat_name_list=[]
        for cat_id in catid_arr:
            if(cat_id=='' or int(cat_id)==0):
                continue
            cat_name_list.append(catid_name[int(cat_id)])
        cat_name_str=",".join(cat_name_list)+line[1]
        goods_data.append([line[0],cat_name_str])
    return goods_data

def close_db():
    db.close()
