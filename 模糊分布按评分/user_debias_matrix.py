import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('P_Matrix_item_diff_5.csv')


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


# 计算每个用户对每个项目的评分与历史加权平均值的差异
diff_results = []

# 分段阶段数
num_segments = 5

# 计算每个分段的加权平均值
for user_id, group in df.groupby('userId'):
    user_ratings = group.values.tolist()
    timestamps = [rating[3] for rating in user_ratings]
    segment_points = np.linspace(min(timestamps), max(timestamps), num_segments + 1)
    segments = [(segment_points[i], segment_points[i + 1]) for i in range(num_segments)]

    for segment in segments:
        total_weighted_sum = 0
        total_membership = 0
        for rating in user_ratings:
            if segment[0] <= rating[3] <= segment[1]:
                membership_degree = trapezoidal_mf(rating[3], segment[0], segment[0] + (segment[1] - segment[0]) * 0.2,
                                                   segment[0] + (segment[1] - segment[0]) * 0.8, segment[1])
                total_weighted_sum += rating[2] * membership_degree
                total_membership += membership_degree
        if total_membership != 0:
            weighted_avg = total_weighted_sum / total_membership
        else:
            weighted_avg = 0
        for rating in user_ratings:
            if segment[0] <= rating[3] <= segment[1]:
                diff = rating[2] - weighted_avg
                # 归一化
                A, B = 1, 1
                normalized_diff = (1 / (1 + np.exp(-A * np.abs(diff) + B)))
                diff_results.append([rating[0], rating[1], normalized_diff])

# 创建DataFrame保存差异结果
diff_df = pd.DataFrame(diff_results, columns=['userId', 'movieId', 'diff'])

# 将结果保存为新的CSV文件
diff_df.to_csv('P_Matrix_user_item_diff_5.csv', index=False)
