
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 参数
    M = 5 # 一致性宽度
    A, B = 1, 1

    a = (pd.read_csv('../datasets/preprocess_100k_500/P_Matrix_item.csv'))\

    # 计算用户自身评价的时点平均值
    # 保存用户时点长度（当前时间节点该用户共有几个评价）
    a['mean'] = ''
    a['num'] = ''
    for index in range(len(a)):
        # 当前的一条数据
        index_info = a[index: index + 1]
        user_id = list(index_info['userId'])[0]

        all_pre_info = a[: index]
        # 查找之前搜索与userId匹配的userId
        all_pre_users = all_pre_info[all_pre_info['userId'] == user_id]
        # 时点长度
        all_pre_user_num = len(list(all_pre_info[all_pre_info["userId"] == user_id]["userId"]))
        curr_rating = float(index_info['rating'])

        a.loc[index, "num"] = all_pre_user_num

        # 若之前的所有数据中不存在当前userId则将mean置为0
        # 否则置为平均值
        if list(all_pre_users[-M:]['rating']) == []:
            a.loc[index, 'mean'] = 0
        else:
            mean_rating = np.mean(all_pre_users[-M:]['rating'])
            a.loc[index, 'mean'] = mean_rating

    # 对用户权值去偏
    a['userId'] = a['userId'].values.astype(int)
    a['movieId'] = a['movieId'].values.astype(int)
    a['mean'] = a['mean'].values.astype(float)
    a["rating"] = (1 / (1 + np.exp(-A * np.abs(a["rating"] - a["mean"]) + B))  )

    # 对去偏后的权值做归一化，这里面的rating是AHR-I的值
    a['rating'] = (1 / (1 + np.exp(-A * np.abs(a["rating"]) + B)))

    # 保存个性化偏好权重矩阵
    a.to_csv('../datasets/preprocess_100k_500/user_item_prefer_weight.csv', index=False)

