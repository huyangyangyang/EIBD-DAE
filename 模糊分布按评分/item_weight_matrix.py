import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('../../datasets/preprocess_1M/new_train_set.csv')

# 计算每个项目的历史平均分
project_avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

# 定义计算差异的函数
def calculate_diff(user_ratings):
    num_segments = 10  # 分段阶段数
    # 定义梯形模糊分布函数
    def trapezoidal_mf(x, start, peak_start, peak_end, end):
        if x <= start or x >= end:
            return 0
        elif peak_start <= x <= peak_end:
            return 1
        elif start < x < peak_start:
            return (x - start) / (peak_start - start)
        else:
            return (end - x) / (end - peak_end)
    # 计算加权平均值
    weighted_avg_ratings = []
    for i in range(num_segments):
        total_weighted_sum = 0
        total_membership = 0
        for rating in user_ratings:
            membership_degree = trapezoidal_mf(rating[3], i * 100, i * 100 + 20, (i + 1) * 100 - 20, (i + 1) * 100)
            item_avg_rating = project_avg_ratings.get(rating[1], 0)  # 获取项目的历史平均分，如果没有，则默认为0
            total_weighted_sum += item_avg_rating * membership_degree
            total_membership += membership_degree
        if total_membership != 0:
            weighted_average = total_weighted_sum / total_membership
        else:
            weighted_average = 0
        weighted_avg_ratings.append(weighted_average)
    # 计算评分与加权平均值之间的差异
    user_item_diff = []
    for rating in user_ratings:
        project_id = rating[1]
        weighted_avg = weighted_avg_ratings[project_id % num_segments]  # 使用项目ID的模来选择加权平均值
        diff = rating[2] - weighted_avg
        # 归一化
        A, B = 1, 1
        normalized_diff =(1 / (1 + np.exp(-A * np.abs(diff) + B)* rating[2]))
        user_item_diff.append([rating[0], project_id, normalized_diff, rating[3]])  # 修改为列表形式，保留时间戳
    return user_item_diff

# 按用户ID和项目ID分组，并应用计算差异的函数
diff_results = df.groupby(['userId', 'movieId']).apply(lambda x: calculate_diff(x.values.tolist())).explode().reset_index(drop=True)

# 创建DataFrame保存差异结果
diff_df = pd.DataFrame(diff_results, columns=['timestamp'])

# 将结果保存为新的CSV文件
diff_df.to_csv('item_prefer_weight_diff_1M.csv', index=False)
