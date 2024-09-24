import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 参数
    A, B = 1, 1

    # 读取数据
    a = pd.read_csv('../../datasets/preprocess_1M/P_Matrix_item_mean.csv')

    # 计算用户的评分平均值
    user_means = a.groupby('userId')['rating'].transform('mean')

    # 替换原来的计算方式，使用用户评分减去用户所有的评分平均值
    a["rating"] = (1 / (1 + np.exp(-A * np.abs(a["rating"] - user_means) + B)))

    # 保存处理后的结果到本地文件
    a.to_csv('../../datasets/preprocess_1M/P_Matrix_user_item_mean.csv', index=False)
