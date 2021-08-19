#!/usr/bin/env python
# encoding=utf-8
import redis
import json

import MySQLdb

r = redis.Redis(host='r-2ze3b63fe630b7c4.redis.rds.aliyuncs.com',port=6379,password='r-2ze3b63fe630b7c4:3R2TJ9y2dkfef6x6',db=0,decode_responses=True)
# print(r.get("common:config:APP:wd_onepercent"))
jsonObj=json.loads(r.get("common:config:APP:wd_onepercent"))
goods_arr=jsonObj["config_value"][0]["value"].split("\n")
# for goods_id in goods_arr:
#     print(goods_id)

# 打开数据库连接
db = MySQLdb.connect(host='wonderfulloffline.mysql.rds.aliyuncs.com',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8' )
db.execute('DELETE FROM goods_pool')

cursor = db.cursor()
try:
    sql = "INSERT INTO EMPLOYEE(goods_id,pool)VALUES('%d','%s')"
    vals=[]
    for goods_id in goods_arr:
        vals.append((goods_id,'px'))
    count=cursor.executemany(sql,vals)#执行sql语句
    db.commit()#提交到数据库执行
    print(count)
except Exception as e:
    db.rollback()
    print(e)
    print('there is some error!')

db.close()