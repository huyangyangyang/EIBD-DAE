
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
    def __init__(self, nb_item, nb_user, nb_hidden1,nb_hidden2, drop_rate=0.3):
        '''
        :param nb_item: 项目的数量
        '''
        super(CDAE, self).__init__()
        self.item2hidden = nn.Sequential(
            nn.Linear(nb_item, nb_hidden1),
            nn.Dropout(drop_rate)
        )
        self.id2 = nn.Sequential(
            nn.Linear(nb_hidden1, nb_hidden2),
            nn.Dropout(drop_rate)
        )
        self.id2hidden = nn.Embedding(nb_user, nb_hidden1)
        self.hidden2out = nn.Linear(nb_hidden2, nb_item)
        self.sigmoid = nn.Sigmoid()

    def forward(self, uid, purchase_vec):
        hidden1 = self.sigmoid(self.id2hidden(uid).squeeze(dim=1)+self.item2hidden(purchase_vec))
        hidden2out = self.id2(hidden1)
        hidden2 = self.sigmoid(hidden2out)
        # hidden = self.sigmoid(self.item2hidden(purchase_vec))
        out = self.hidden2out(hidden2)
        out = self.sigmoid(out)
        return out