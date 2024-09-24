import torch
import torch.nn as nn
import random
import pandas as pd
import torch
import numpy as np
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
np.random.seed(4)
random.seed(4)
torch.backends.cudnn.deterministic = True


class CDAE(nn.Module):
    def __init__(self, nb_item, nb_user, nb_hidden1, nb_hidden2, drop_rate=0.3, anchor_bias=None):
        super(CDAE, self).__init__()
        self.nb_item = nb_item
        self.nb_user = nb_user
        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2

        # 定义锚定一致性偏差节点的权重矩阵
        self.anchor_bias_weight = nn.Parameter(torch.randn(nb_user, nb_hidden1))

        # 定义第一个隐藏层的权重矩阵
        self.item2hidden = nn.Sequential(
            nn.Linear(nb_item, nb_hidden1),
            nn.Dropout(drop_rate)
        )

        # 其他层的定义保持不变
        self.id2 = nn.Sequential(
            nn.Linear(nb_hidden1, nb_hidden2),
            nn.Dropout(drop_rate)
        )
        self.id2hidden = nn.Embedding(nb_user, nb_hidden1)
        self.hidden2out = nn.Linear(nb_hidden2, nb_item)
        self.sigmoid = nn.Sigmoid()

        # 初始化锚定一致性偏差节点的权重矩阵
        if anchor_bias is not None:
            self.anchor_bias_weight.data.copy_(torch.tensor(anchor_bias))

    def forward(self, uid, purchase_vec):
        # 计算锚定一致性偏差节点的输出
        anchor_bias_output = torch.matmul(uid, self.anchor_bias_weight)

        # 计算第一个隐藏层的输出
        hidden1 = self.sigmoid(self.id2hidden(uid).squeeze(dim=1) + self.item2hidden(purchase_vec) + anchor_bias_output)

        # 其他层的计算保持不变
        hidden2out = self.id2(hidden1)
        hidden2 = self.sigmoid(hidden2out)
        out = self.hidden2out(hidden2)
        out = self.sigmoid(out)
        return out
