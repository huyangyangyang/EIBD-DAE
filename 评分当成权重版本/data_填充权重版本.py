

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random


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
