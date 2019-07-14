# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:34:27 2019

@author: Administrator
"""

import execjs


js = """
function add(x, y){
    return x + y;
}
"""
ctx = execjs.compile(js)
a=ctx.call("add", 3, 4)
print(a)
print(execjs.get().name)