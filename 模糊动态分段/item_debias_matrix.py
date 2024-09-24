import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('your_file.csv')

# 计算每个项目的历史平均分
project_avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

# 定义梯形模糊逻辑函数
def trapezoidal_mf(x, start, peak_start, peak_end, end):
    if x <= start or x >= end:
        return 0
    elif peak_start <= x <= peak_end:
        return 1
    elif start < x < peak_start:
        return (x - start) / (peak_start - start)
    else:
        return (end - x) / (end - peak_end)

# 定义计算加权平均值的函数
def calculate_weighted_avg(user_ratings, num_segments):
    segment_points = np.linspace(user_ratings['timestamp'].min(), user_ratings['timestamp'].max(), num_segments + 1)
    segments = [(segment_points[i], segment_points[i+1]) for i in range(num_segments)]
    weighted_avg_ratings = []
    for segment in segments:
        total_weighted_sum = 0
        total_membership = 0
        for rating in user_ratings.itertuples():
            if segment[0] <= rating.timestamp <= segment[1]:
                membership_degree = trapezoidal_mf(rating.timestamp, segment[0], segment[0] + (segment[1] - segment[0]) * 0.2, segment[0] + (segment[1] - segment[0]) * 0.8, segment[1])
                item_avg_rating = project_avg_ratings.get(rating.movieId, 0)  # 获取项目的历史平均分，如果没有，则默认为0
                total_weighted_sum += item_avg_rating * membership_degree
                total_membership += membership_degree
        if total_membership != 0:
            weighted_average = total_weighted_sum / total_membership
        else:
            weighted_average = 0
        weighted_avg_ratings.append(weighted_average)
    return weighted_avg_ratings

# 定义归一化函数
def normalize_diff(diff):
    A, B = 1, 1
    return 5 * (1 / (1 + np.exp(-A * np.abs(diff) + B)))

# 对每个用户进行计算加权平均值和归一化处理
user_normalized_diffs = []
for user_id, user_ratings in df.groupby('userId'):
    num_segments = 10  # 可根据用户打分数量进行动态调整
    if len(user_ratings) > 0:
        weighted_avg_ratings = calculate_weighted_avg(user_ratings, num_segments)
        for rating in user_ratings.itertuples():
            segment_index = int((rating.timestamp - user_ratings['timestamp'].min()) / (user_ratings['timestamp'].max() - user_ratings['timestamp'].min()) * num_segments)
            weighted_avg = weighted_avg_ratings[segment_index]
            diff = rating.rating - weighted_avg
            normalized_diff = normalize_diff(diff)
            user_normalized_diffs.append([rating.userId, rating.movieId, normalized_diff, rating.timestamp])

# 创建DataFrame保存归一化差异结果
diff_df = pd.DataFrame(user_normalized_diffs, columns=['userId', 'movieId', 'normalized_diff', 'timestamp'])

# 将结果保存为新的CSV文件
diff_df.to_csv('normalized_diff_result.csv', index=False)
