
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 98765


# 切分训练集和测试集
data = pd.read_csv("../datasets/preprocess_1M_0.9/1M_0.9.csv")
train_set, test_set = train_test_split(data, test_size=0.2, random_state=SEED)

# 保存训练集数据
train_set = train_set.reset_index(drop=True)
# new_train_set为ID为自定义ID的训练集
train_set.to_csv('../datasets/preprocess_1M_0.9/train_set.csv', index=False)

# 保存测试集数据
test_set = test_set.reset_index(drop=True)
# new_test_set的ID未自定义ID的测试集
test_set.to_csv('../datasets/preprocess_1M_0.9/test_set.csv', index=False)

# 将测试集中小于平均权值的关键词消除
a = pd.read_csv('../datasets/preprocess_1M_0.9/test_set.csv')
mean_rating = np.mean(pd.read_csv('../datasets/preprocess_1M_0.9/1M_0.9.csv')['rating'])
a = a[a['rating'] > mean_rating]
a.to_csv('../datasets/preprocess_1M_0.9/test_set_filtered.csv', index=False)

