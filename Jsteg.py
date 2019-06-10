# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:26:43 2019

@author: Administrator
"""

import math
 
class Jsteg:
    def __init__(self):
        self.sequence_after_dct=None
 
    def set_sequence_after_dct(self,sequence_after_dct):
        self.sequence_after_dct=sequence_after_dct
        self.available_info_len=len([i for i in self.sequence_after_dct if i not in (-1,1,0)]) # 不是绝对可靠的
        print ("Load>> 可嵌入",self.available_info_len,'bits')
     
    def get_sequence_after_dct(self):
        return self.sequence_after_dct
 
    def write(self,info):
        """先嵌入信息的长度，然后嵌入信息"""
        info=self._set_info_len(info)
        info_len=len(info)
        info_index=0
        im_index=0
        while True:
            if info_index>=info_len:
                break
            data=info[info_index]
            if self._write(im_index,data):
                info_index+=1
            im_index+=1
 
 
    def read(self):
        """先读出信息的长度，然后读出信息"""
        _len,sequence_index=self._get_info_len()
        info=[]
        info_index=0
 
        while True:
            if info_index>=_len:
                break
            data=self._read(sequence_index)
            if data!=None:
                info.append(data)
                info_index+=1
            sequence_index+=1
 
        return info
 
    #===============================================================#
 
    def _set_info_len(self,info):
        l=int(math.log(self.available_info_len,2))+1
        info_len=[0]*l
        _len=len(info)
        info_len[-len(bin(_len))+2:]=[int(i) for i in bin(_len)[2:]]
        return info_len+info
 
    def _get_info_len(self):
        l=int(math.log(self.available_info_len,2))+1
        len_list=[]
        _l_index=0
        _seq_index=0
        while True:
            if _l_index>=l:
                break
            _d=self._read(_seq_index)
            if _d!=None:
                len_list.append(str(_d))
                _l_index+=1
            _seq_index+=1
        _len=''.join(len_list)
        _len=int(_len,2)
        return _len,_seq_index
    
    # 注意经过DCT会有负值，此处最低有效位的嵌入方式与空域LSB略有不同
    def _write(self,index,data):
        origin=self.sequence_after_dct[index]
        if origin in (-1,1,0):
            return False
 
        lower_bit=origin%2
        if lower_bit==data:
            pass
        elif origin>0:
            if (lower_bit,data) == (0,1):
                self.sequence_after_dct[index]=origin+1
            elif (lower_bit,data) == (1,0):
                self.sequence_after_dct[index]=origin-1
        elif origin<0:
            if (lower_bit,data) == (0,1):
                self.sequence_after_dct[index]=origin-1
            elif (lower_bit,data) == (1,0):
                self.sequence_after_dct[index]=origin+1
 
        return True
 
    def _read(self,index):
        if self.sequence_after_dct[index] not in (-1,1,0):
            return self.sequence_after_dct[index]%2
        else:
            return None
'''
import cv2
import numpy as np
 
def dct(m):
    m = np.float32(m)/255.0
    return cv2.dct(m)*255
'''

if __name__=="__main__":
    jsteg=Jsteg()
    # 写
    sequence_after_dct=[-1,0,1]*100+[i for i in range(-7,500)]
    #print(sequence_after_dct)
    jsteg.set_sequence_after_dct(sequence_after_dct)
    info1=[0,1,0,1,1,0,1,0]
    jsteg.write(info1)
    sequence_after_dct2=jsteg.get_sequence_after_dct()
    # 读
    jsteg.set_sequence_after_dct(sequence_after_dct2)
    info2=jsteg.read()
    print (info2)