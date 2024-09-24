
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
# 上述代码的功能是实现从ml-1m数据集的三个文件中读取数据，分别是训练集、测试集和匹配集，
# 然后将它们转换成矩阵形式。其中，读取数据的函数为read_ml1m()，
# 该函数返回三个字典类型的数据，分别对应三个数据集。转换成矩阵形式的函数为get_matrix()，
# 该函数根据用户数量和电影数量生成对应维度的全0矩阵，然后将训练集、测试集和匹配集中的评分数据填充到矩阵中。
# 其中，匹配集中的评分为0的位置会被填充成一个随机的小数。这是为了表示用户和该电影之间没有评分记录，但仍然具有一定的相关性。

def read_ml1m(filepath1, filepath2, filepath3):
    train_set_dict, test_set_dict, nihe_dict = {}, {}, {}
    df = pd.read_csv(filepath1).iloc[:, :3] - 1

    df = df.values.tolist()
    df2 = pd.read_csv(filepath2).iloc[:, :3] - 1
    df2 = df2.values.tolist()
    df3 = pd.read_csv(filepath3).iloc[:, :3] - 1
    df3 = df3.values.tolist()

    train_set, test_set, nihe_set = df, df2, df3
    for uid, iid, score in train_set:
        uid = int(uid)
        iid = int(iid)
        train_set_dict.setdefault(uid, {}).setdefault(iid, round(score + 1, 5))
    for uid, iid, score in test_set:
        uid = int(uid)
        iid = int(iid)
        test_set_dict.setdefault(uid, {}).setdefault(iid, round(score + 1, 5))
    for uid, iid, score in nihe_set:
        uid = int(uid)
        iid = int(iid)
        # print(score+1)
        nihe_dict.setdefault(uid, {}).setdefault(iid, round(score + 1, 5))

    return train_set_dict, test_set_dict, nihe_dict


def get_matrix(train_set_dict, test_set_dict, nihe_dict, nb_user, nb_item):
    train_set, test_set, nihe_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item)), np.zeros(
        shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = train_set_dict[u][i]
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = test_set_dict[u][i]
    for u in nihe_dict.keys():
        for i in nihe_dict[u]:
            nihe_set[u][i] = nihe_dict[u][i]

    #
    for i in range(nihe_set.shape[0]):
        # 得到所有不为0的项目下标
        # 这些位置的值为1
        items2 = np.where(nihe_set[i] == 0)[0].tolist()
        for iii in items2:
            a = random.random()
            nihe_set[i][iii] = a + 1

    return train_set, test_set, nihe_set
