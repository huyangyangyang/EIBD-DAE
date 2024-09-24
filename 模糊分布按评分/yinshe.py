import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import heapq
import sys

def train_test_ml(data):
    # ID值映射
    uid2internal, internal2uid = dict(), dict()
    iid2internal, internal2iid = dict(), dict()
    uid_list = data['user_id'].unique()
    iid_list = data['item_id'].unique()
    for i, uid in enumerate(uid_list):
        uid2internal[int(uid)] = i + 1
        internal2uid[i + 1] = int(uid)
    for i, iid in enumerate(iid_list):
        iid2internal[int(iid)] = i + 1
        internal2iid[i + 1] = int(iid)
    data['user_id'].replace(uid2internal, inplace=True)
    data['item_id'].replace(iid2internal, inplace=True)
    return data

