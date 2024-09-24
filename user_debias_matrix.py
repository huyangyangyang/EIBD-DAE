

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 参数
    M = 5 # 一致性宽度
    A, B = 1, 1

    a = (pd.read_csv('../datasets/preprocess_100k_500/P_Matrix_item.csv'))

    # 计算用户评价的时点平均值
    # 保存用户时点长度（当前时间节点该用户共评价过几个电影）
    a['mean'] = ''
    a['num'] = ''
    for index in range(len(a)):
        # 当前的一条数据
        index_info = a[index: index + 1]
        user_id = list(index_info['userId'])[0]

        all_pre_info = a[: index]
        # 查找之前评价过的电影
        all_pre_movies = all_pre_info[all_pre_info['userId'] == user_id]
        # 时点长度
        all_pre_movie_num = len(list(all_pre_info[all_pre_info["userId"] == user_id]["userId"]))
        curr_rating = float(index_info['rating'])

        a.loc[index, "num"] = all_pre_movie_num

        # 若之前的所有数据中不存在当前userId则将mean置为0
        # 否则置为平均值
        if list(all_pre_movies[-M:]['rating']) == []:
            a.loc[index, 'mean'] = 0
        else:
            mean_rating = np.mean(all_pre_movies[-M:]['rating'])
            a.loc[index, 'mean'] = mean_rating

    # 对用户评价的时点平均值进行去偏
    a['userId'] = a['userId'].values.astype(int)
    a['movieId'] = a['movieId'].values.astype(int)
    a['mean'] = a['mean'].values.astype(float)

    # 构建去偏权值矩阵P, 并归一化
    a["rating"] = (1 / (1 + np.exp(-A * np.abs(a["rating"] - a["mean"]) + B)))

    # 保存去偏后的P矩阵到本地文件
    del a['mean']
    a.to_csv('../datasets/preprocess_100k_500/P_Matrix_user_item.csv', index=False)
