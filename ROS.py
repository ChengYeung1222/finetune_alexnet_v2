import random
from random import randint
import numpy as np

oldf_r = open('./dygz_train_20c_list.csv', 'r', encoding='UTF-8')
oldf = './dygz_train_20c_list.csv'
newf = open('./dygz_train_20c_list_ROS.csv', 'w', encoding='UTF-8')
# oldf = open('./List/dygz_train_20c_list_10_percent.csv', 'r', encoding='UTF-8')
# newf = open('./List/dygz_train_20c_list_128.csv', 'w', encoding='UTF-8')
# n = 0

# resultList = random.sample(range(0, 2155), 128)

deleted_zero_list = []
one_list = []
with open(oldf, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        items = line.split(',')
        if int(items[1]) == 0:
            # print(i)
            deleted_zero_list.append(i)
        elif int(items[1]) == 1:
            one_list.append(i)

deleted_zero_list_length, one_list_length = len(deleted_zero_list), len(one_list)
print(deleted_zero_list_length)
print(one_list_length)

resultList_one = np.random.choice(one_list, size=deleted_zero_list_length, replace=True)
# resultList_zero = random.sample(deleted_zero_list, deleted_zero_list_length // 10)
resultList_zero = np.array(deleted_zero_list)
resultList = np.concatenate((resultList_zero, resultList_one))
random.shuffle(resultList)
lines = oldf_r.readlines()
#
for i in resultList:
    newf.write(lines[i])
#
# # oldf.close()
newf.close()
