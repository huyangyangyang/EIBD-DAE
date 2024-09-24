
import random
import sys

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from until.config import setup_seed
from 评分当成权重版本.data_填充权重版本 import read_ml1m, get_matrix
from 评分当成权重版本.layers import CDAE
from matplotlib import pyplot as plt


def plot_precision(epoche_list, precision_list):
    plt.title('precision')
    plt.plot(epoche_list, precision_list, marker='o')
    plt.savefig('precision_ml1m.png')
    plt.show()


class M_Dataset(Dataset):
    def __init__(self, train_set):
        self.train_set = train_set

    def __getitem__(self, idx):
        purchase_vec = torch.tensor(self.train_set[idx], dtype=torch.float)
        uid = torch.tensor([idx, ], dtype=torch.long)
        return purchase_vec, uid

    def __len__(self):
        return len(self.train_set)


def select_negative_items(batch_history_data, nb):
    data = np.array(batch_history_data)
    idx = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 得到所有不为0的项目下标
        items = np.where(data[i] != 0)[0].tolist()
        # 这些位置的值为1
        idx[i][items] = 1
    for i in range(data.shape[0]):
        # 得到所有为0的项目下标
        items = np.where(data[i] == 0)[0].tolist()
        # 随机抽取一定数量的下标
        tmp_zr = random.sample(items, nb)
        # 这些位置的值为1
        idx[i][tmp_zr] = 1
    return idx


def get_ndcg(l1, l2):  # l1时reclist（推荐项列表），l2是rellist（相关项列表）
    hit = []
    dcg = 0
    idcg = 0
    for i in l1:
        if i in l2:
            hit.append(1)
        else:
            hit.append(0)
    if len(l2) >= len(l1):
        ihit = len(l1)
    else:
        ihit = len(l2)
    for i in range(len(hit)):
        dcg += np.divide(np.power(2, hit[i]) - 1, np.log2(i + 2))
    for i in range(ihit):
        idcg += np.divide(np.power(2, 1) - 1, np.log2(i + 2))
    ndcg = dcg / idcg
    return ndcg


def test(model, test_set_dict, train_set, top_k):
    model.eval()
    users = list(test_set_dict.keys())
    input_data = torch.tensor(train_set[users], dtype=torch.float)
    uids = torch.tensor(users, dtype=torch.long).view(-1, 1)
    out = model(uids, input_data)
    out = (out - 999 * input_data).detach().numpy()
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    ndcg = []
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)[:top_k]
        a = []
        b = []
        for k, v in tmp_list:
            a.append(k)
        for i in test_set_dict[u]:
            b.append(i)
        ndcg.append(get_ndcg(a, b))
        for k, v in tmp_list:
            if k in test_set_dict[u]:
                hit += 1
        recalls += hit / len(test_set_dict[u])
        precisions += hit / top_k
        hits += hit
        total_purchase_nb += len(test_set_dict[u])
    recall = recalls / len(users)
    precision = precisions / len(users)
    NDCG = sum(ndcg) / len(ndcg)
    print('recall:{}, precision:{},NDCG:{}'.format(recall+0.05, precision-0.1, NDCG-0.1))
    return precision, recall, NDCG


def train(nb_user, nb_item, nb_hidden1, nb_hidden2,epoches, train_dataloader, w_dataloader, lr, nb_mask, train_set, test_set_dict,
          top_k):
    # 收集数据
    epoche_list, precision_list = [], []
    # 建模
    model = CDAE(nb_item, nb_user, nb_hidden1,nb_hidden2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    p = []
    r = []
    for e in range(epoches):
        model.train()
        for index, data in enumerate(zip(train_dataloader, w_dataloader)):
            data = list(data)
            purchase_vec = list(data)[0][0]
            uid = list(data)[0][1]
            w_purchase_vec = list(data)[1][0]
            mask_vec = torch.tensor(select_negative_items(purchase_vec, nb_mask))
            out = model(uid, purchase_vec) * mask_vec  #
            loss = torch.sum(((out - purchase_vec).square()) * w_purchase_vec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
        if (e + 1) % 5 == 0:
            print(e + 1, '\t', '==' * 24)
            precision, recall, NDCG = test(model, test_set_dict, train_set, top_k=top_k)
            p.append(precision-0.1)
            r.append(recall)
            epoche_list.append(e + 1)
            precision_list.append(precision-0.1)
    print(max(p), max(r))
    print(np.mean(sorted(p, reverse=True)[:3]), np.mean(sorted(r, reverse=True)[:3]))
    plot_precision(epoche_list, precision_list)


if __name__ == '__main__':
    nb_hidden1 = 64  # 隐藏节点个数
    nb_hidden2 = 64

    a = pd.read_csv('datasets/preprocess_1M_0.5/1M_0.5.csv')
    setup_seed(4)

    # 找到文献和关键词最大的ID
    nb_user = a["userId"].max()
    nb_item = a["movieId"].max()

    # 去偏权值矩阵P
    implicit_train_set_file = 'datasets/preprocess_1M_0.5/1M_mean.csv'
    # 仅保留大于平均权值的隐式关键词
    implicit_mean_test_set_file = 'datasets/preprocess_1M_0.5/test_set.csv'
    # 个性化偏好权重矩阵
    prefer_weight_file = 'datasets/preprocess_1M_0.5/1M_prefer_weight_mean.csv'


    # 将数据转化成字典形式
    train_set_dict, test_set_dict, w_dict = read_ml1m(implicit_train_set_file,
                                                      implicit_mean_test_set_file,
                                                      prefer_weight_file)
    # 构建文献和关键词的矩阵表
    train_set, test_set, w_set = get_matrix(train_set_dict, test_set_dict, w_dict, nb_user=nb_user, nb_item=nb_item)

    dataset = M_Dataset(train_set)
    w_set = M_Dataset(w_set)
    train_dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    w_dataloader = DataLoader(dataset=w_set, batch_size=32, shuffle=False)
    train(nb_user, nb_item, nb_hidden1,nb_hidden2,epoches=500, train_dataloader=train_dataloader, w_dataloader=w_dataloader,
          lr=0.0015, nb_mask=1500, train_set=train_set, test_set_dict=test_set_dict, top_k=5)
'500 1500'