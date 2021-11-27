
import random
from random import randint

oldf =open('./List/dygz_train_20c_list_10_percent.csv' ,'r' ,encoding='UTF-8')
newf =open('./List/dygz_train_20c_list_128.csv' ,'w' ,encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0 ,2155) ,128)

lines =oldf.readlines()
for i in resultList:
    newf.write(lines[i])

oldf.close()
newf.close()

