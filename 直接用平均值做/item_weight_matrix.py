import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 参数
    A, B = 1, 1

    # 读取数据
    data = pd.read_csv('../../datasets/amazon_0.5/train_set.csv')
    del data['timestamp']

    # 计算每个项目的历史评分平均值
    project_means = data.groupby('movieId')['rating'].transform('mean')

    # 对关键字权值去偏
    data['rating'] = 5 * (1 / (1 + np.exp(-A * np.abs(data['rating'] - project_means) + B)* data['rating']))

    # 保存结果
    data.to_csv('../../datasets/amazon_0.5/amazon_prefer_weight_mean.csv', index=False)
