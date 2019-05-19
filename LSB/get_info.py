# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:43:26 2019

@author: Administrator
"""

from PIL import Image

def mod(x,y): 
    return x%y

def toasc(strr):
    return int(strr, 2)
        
         
#le为所要提取的信息的长度，str1为加密载体图片的路径，str2为提取文件的保存路径
def func(le,str1,str2): 
    b="" 
    im = Image.open(str1)
    lenth = le*8
    width,height = im.size[0],im.size[1]
    count = 0
    for h in range(height): 
        for w in range(width):
            #获得(w,h)点像素的值
            pixel = im.getpixel((w, h))
            #此处余3，依次从R、G、B三个颜色通道获得最低位的隐藏信息 
            if count%3==0:
                count+=1 
                b=b+str((mod(int(pixel[0]),2))) 
                if count ==lenth:
                    break
            if count%3==1:
                count+=1
                b=b+str((mod(int(pixel[1]),2)))
                if count ==lenth:
                    break
            if count%3==2: 
                count+=1
                b=b+str((mod(int(pixel[2]),2)))
                if count ==lenth:
                    break
        if count == lenth:
            break
 
    with open(str2,"w",encoding='utf-8') as f: 
        for i in range(0,len(b),8):
            #以每8位为一组二进制，转换为十进制            
            stra = toasc(b[i:i+8]) 
            #将转换后的十进制数视为ascii码，再转换为字符串写入到文件中
            #print((stra))
            f.write(chr(stra))
    print("完成信息提取！")




def main():   
    #文件长度 
    le = 11
    #含有隐藏信息的图片 
    new = "new.png" 
    #信息提取出后所存放的文件
    get_info = "get_flag.txt"
    func(le,new,get_info)
    
if __name__=='__main__':
    main()