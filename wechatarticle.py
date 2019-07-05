# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:51:21 2019

@author: Administrator
"""

import requests
import json
import time
from pymongo import MongoClient

url = 'http://mp.weixin.qq.com/mp/profile_ext'  #（公众号不让添加主页链接，xxx表示profile_ext)

# Mongo配置
conn = MongoClient('127.0.0.1', 27017)
db = conn.wx  #连接wx数据库，没有则自动创建
mongo_wx = db.article  #使用article集合，没有则自动创建

def get_wx_article(biz, uin, key,index=0, count=10):
    offset = 1+(index + 1) * 11
    params = {
        '__biz': biz,
        'uin': uin,
        'key': key,
        'offset': offset,
        'count': count,
        'action': 'getmsg',
        'f': 'json',
        'pass_ticket':'c+5vPIVHiNoUPbaMDdjgNChWAKjNBWHu0aJKMpE8DjuM8Sz+JVstUGkhmQybwPFJ',
        'scene':124,
        'is_ok':1,
        'appmsg_token':'1016_XyCOVNsuNMHk9dKFYO2FSsVMDsB9HP_hzROzAg~~',
        'x5':0,
        
        
        
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    }

    response = requests.get(url=url, params=params, headers=headers)
    resp_json = response.json()
    print(resp_json)
    if resp_json.get('errmsg') == 'ok':
        resp_json = response.json()
        # 是否还有分页数据， 用于判断return的值
        can_msg_continue = resp_json['can_msg_continue']
        # 当前分页文章数
        msg_count = resp_json['msg_count']
        general_msg_list = json.loads(resp_json['general_msg_list'])
        list = general_msg_list.get('list')
        print(list, "**************")
        for i in list:
            app_msg_ext_info = i['app_msg_ext_info']
            # 标题
            title = app_msg_ext_info['title']
            # 文章地址
            content_url = app_msg_ext_info['content_url']
            # 封面图
            cover = app_msg_ext_info['cover']

            # 发布时间
            datetime = i['comm_msg_info']['datetime']
            datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(datetime))

            mongo_wx.insert({
                'title': title,
                'content_url': content_url,
                'cover': cover,
                'datetime': datetime
            })
        if can_msg_continue == 1:
            return True
        return False
    else:
        print('获取文章异常...')
        return False


if __name__ == '__main__':
    biz = 'MzA5NDk4NDcwMw=='
    uin = 'MTY1NTcyMDYxMg=='
    key = '4af267131119af887b17806c8f5467404ade6a90a1f90a5f3fd05c32bd0121eeb326d246b40f975942600824bcb6859403126fc59a19bae103a45247b08635b1100f3c737c4c785b1fa70d4a41d11991'
    index = 0
    while 1:
        print(f'开始抓取公众号第{index + 1} 页文章.')
        flag = get_wx_article(biz, uin, key, index=index)
        # 防止和谐，暂停8秒
        time.sleep(8)
        index += 1
        if not flag:
            print('公众号文章已全部抓取完毕，退出程序.')
            break

        print(f'..........准备抓取公众号第{index + 1} 页文章.')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        