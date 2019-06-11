# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:30:40 2019

@author: Administrator
"""

from Jsteg import Jsteg

 
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