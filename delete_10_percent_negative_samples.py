import random
from random import randint

oldf_r = open('./preResoult_20c_dygz_26940.csv', 'r', encoding='UTF-8')
oldf = './preResoult_20c_dygz_26940.csv'
newf = open('./preResoult_20c_dygz_26940_delete_negative.csv', 'w', encoding='UTF-8')
# oldf = open('./List/dygz_train_20c_list_10_percent.csv', 'r', encoding='UTF-8')
# newf = open('./List/dygz_train_20c_list_128.csv', 'w', encoding='UTF-8')
# n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0, 2155), 128)

deleted_zero_list = []
one_list = []
with open(oldf, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        items = line.split(',')
        if int(items[0]) == 0:
            # print(i)
            deleted_zero_list.append(i)
        elif int(items[0]) == 1:
            one_list.append(i)

deleted_zero_list_length = len(deleted_zero_list)
print(deleted_zero_list_length)
resultList_zero = random.sample(deleted_zero_list, deleted_zero_list_length // 10)
resultList = resultList_zero + one_list

lines = oldf_r.readlines()
#
for i in resultList:
    newf.write(lines[i])
#
# # oldf.close()
newf.close()
