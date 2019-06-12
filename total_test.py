# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:36:29 2019

@author: Administrator
"""

import math
import numpy as np
import DCT

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
        #print(self.sequence_after_dct)
        b={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,-1:0,-2:0,-3:0,-4:0,-5:0,-6:0,-7:0,-8:0}
        for i in self.sequence_after_dct:
            if i in b:
                b[i]+=1
        print("经过信息隐藏后JPEG的DCT系数变化")
        print(b)
 
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
        #print(info_len+info)
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
        origin=self.sequence_after_dct[index]
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
        
        
class F4(Jsteg):
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
        origin=self.sequence_after_dct[index]
        if origin == 0:
            return False
        elif origin == 1 and data==0:
            self.sequence_after_dct[index]=0
            return False
        
        elif origin == -1 and data==1:
            self.sequence_after_dct[index]=0
            return False
        
        lower_bit=origin%2
        
        if origin >0:
            if lower_bit!=data:
                self.sequence_after_dct[index]=origin-1
        else:
            if lower_bit==data:
                self.sequence_after_dct[index]=origin+1
        return True

 
    def _read(self,index):
        if self.sequence_after_dct[index] >0:
            return self.sequence_after_dct[index]%2
        elif self.sequence_after_dct[index]<0:
            return (self.sequence_after_dct[index]+1)%2
        else:
            return None




if __name__=="__main__":
    jsteg=Jsteg()
    f3=F3()
    f4=F4()
    # 写
    sequence_after_dct=DCT.after_dct
    jsteg.set_sequence_after_dct(sequence_after_dct)
    f3.set_sequence_after_dct(sequence_after_dct)
    f4.set_sequence_after_dct(sequence_after_dct)
    a={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,-1:0,-2:0,-3:0,-4:0,-5:0,-6:0,-7:0,-8:0}
    for i in sequence_after_dct:
        if i in a:
            a[i]+=1
    print("JPEG的DCT系数")
    print(a)
    info1=[int(i+0.5) for i in np.random.rand(200000)]
    
    print("Jsteg begin writing!")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    jsteg.write(info1)

    print("F3steg begin writing!")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    f3.write(info1)
    print("F4steg begin writing!")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    f4.write(info1)
    '''
    # 读
    sequence_after_dct2=jsteg.get_sequence_after_dct()
    sequence_after_dct3=f3.get_sequence_after_dct()
    sequence_after_dct4=f4.get_sequence_after_dct()
    jsteg.set_sequence_after_dct(sequence_after_dct2)
    f3.set_sequence_after_dct(sequence_after_dct3)
    f4.set_sequence_after_dct(sequence_after_dct4)
    info2=jsteg.read()
    info3=f3.read()
    info4=f4.read()
    '''
