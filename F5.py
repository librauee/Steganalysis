# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:05:46 2019

@author: Administrator
"""

from Jsteg import Jsteg
import numpy as np


'''
k= 3， 则 ：n = 7 有 
a1,a2,a3,a4,a5,a6,a7，x1,x2,x3 
假设: 
a1 = 1 
a2 = 1 
a3 = 0 
a4 = 1 
a5 = 0 
a6 = 0 
a7 = 1 
x1 = 1 
x2 = 1 
x3 = 0 
f(a) = 1*1 ⊕ 1*2 ⊕ 1*4 ⊕ 1*7 = 001 ⊕ 010 ⊕ 100 ⊕ 111 = 000 = 0

s = f(a) ⊕ x = 000 ⊕ 110 = 110 = 6 
则改变a6为1即可完成编码嵌入x1,x2,x3 
提取方法： 
f(a’) = 1*1 ⊕ 1*2 ⊕ 1*4 ⊕ 1*6 ⊕ 1*7 = 001 ⊕ 010 ⊕ 100 ⊕ 110 ⊕ 111 = 110 
即x1,x2,x3分别为1,1,0，提取正确 
若s = 0, 则不用修改就可以实现嵌入；

'''

 
class F3(Jsteg):
    def __init__(self):
        Jsteg.__init__(self)
 
    def set_sequence_after_dct(self,sequence_after_dct):
        self.sequence_after_dct=sequence_after_dct
        sum_len=len(self.sequence_after_dct)
        zero_len=len([i for i in self.sequence_after_dct if i==0])
        one_len=len([i for i in self.sequence_after_dct if i in (-1,1)])
        self.available_info_len=sum_len-zero_len-one_len # 不是特别可靠
        print ("Load>> 大约可嵌入",sum_len-zero_len-int(one_len/2),'bits')
        print ("Load>> 最少可嵌入",self.available_info_len,'bits\n')
 
    def _write(self,index,data):
        origin=self.sequence_after_dct[index:index+n]
        f=0
        for i in n-1:
            if origin[i]==1:
                f^=i
        sumdata=0
        for i in k:
            sumdata+=data[i]*(2**i)

        s=f^sumdata
        origin[s-1]=1 if origin[s-1]==0 else 0
        
        
        
        if origin == 0:
            return False
        elif origin in (-1,1) and data==0:
            self.sequence_after_dct[index]=0
            return False
 
        lower_bit=origin%2
 
        if lower_bit==data:
            pass
        elif origin>0:
            self.sequence_after_dct[index]=origin-1
        elif origin<0:
            self.sequence_after_dct[index]=origin+1
        return True
 
    def _read(self,index):
        if self.sequence_after_dct[index] != 0:
            return self.sequence_after_dct[index]%2
        else:
            return None
 
 
if __name__=="__main__":
    f3=F3()
    # 写
    sequence_after_dct=[-1,0,1]*100+[i for i in range(-7,500)]
    f3.set_sequence_after_dct(sequence_after_dct)
    info1=[0,1,0,1,1,0,1,0]
    f3.write(info1)
    # 读
    sequence_after_dct2=f3.get_sequence_after_dct()    
    f3.set_sequence_after_dct(sequence_after_dct2)
    info2=f3.read()
    print (info2)