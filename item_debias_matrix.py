
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 参数
    M = 5 # 一致性宽度
    A, B = 1, 1

    a = (pd.read_csv('../datasets/preprocess_1M_0.5/train_set.csv')).sort_values('timestamp').reset_index(drop=True)
    del a['timestamp']

    # 计算关键词权值的时点平均权值
    # 保存关键词时点长度（当前时间节点该关键词共出现过几次）
    a['mean'] = ''
    a['num'] = ''
    for index in range(len(a)):
        # 当前的一条数据
        index_info = a[index: index + 1]
        movie_id = list(index_info['movieId'])[0]

        all_pre_info = a[: index]
        # 查找之前搜索与movieId匹配的movieId
        all_pre_movies = all_pre_info[all_pre_info['movieId'] == movie_id]
        # 时点长度
        all_pre_movie_num = len(list(all_pre_info[all_pre_info["movieId"] == movie_id]["movieId"]))
        curr_rating = float(index_info['rating'])

        a.loc[index, "num"] = all_pre_movie_num

        # 若之前的所有数据中不存在当前movieId则将mean置为0
        # 否则置为平均值
        if list(all_pre_movies[-M:]['rating']) == []:
            a.loc[index, 'mean'] = 0
        else:
            mean_rating = np.mean(all_pre_movies[-M:]['rating'])
            a.loc[index, 'mean'] = mean_rating

    # TODO:可删除
    # a.to_csv('test.csv', index=False)
    # a = pd.read_csv('test.csv')

    # 对关键字权值去偏
    a['userId'] = a['userId'].values.astype(int)
    a['movieId'] = a['movieId'].values.astype(int)
    a['mean'] = a['mean'].values.astype(float)

    # 构建去偏权值矩阵P, 并归一化
    a["rating"] = (1 / (1 + np.exp(-A * np.abs(a["rating"] - a["mean"]) + B)))

    # 保存个性化偏好权重矩阵
    del a['mean']
    a.to_csv('../datasets/preprocess_1M_0.5/P_Matrix_item.csv', index=False)

