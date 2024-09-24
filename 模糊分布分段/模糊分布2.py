import pandas as pd
import numpy as np


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


# 定义计算差异的函数
def calculate_diff(user_ratings, project_avg_ratings):
    num_segments = 10  # 分段阶段数
    weighted_avg_ratings = []
    for i in range(num_segments):
        total_weighted_sum = 0
        total_membership = 0
        for rating in user_ratings:
            membership_degree = trapezoidal_mf(rating['timestamp'], i * 100, i * 100 + 20, (i + 1) * 100 - 20,
                                               (i + 1) * 100)
            item_avg_rating = project_avg_ratings.get(rating['movieId'], 0)
            total_weighted_sum += item_avg_rating * membership_degree
            total_membership += membership_degree
        if total_membership != 0:
            weighted_average = total_weighted_sum / total_membership
        else:
            weighted_average = 0
        weighted_avg_ratings.append(weighted_average)

    user_item_diff = []
    for rating in user_ratings:
        project_id = rating['movieId']
        weighted_avg = weighted_avg_ratings[project_id % num_segments]
        diff = rating['rating'] - weighted_avg
        A, B = 1, 1
        normalized_diff = (1 / (1 + np.exp(-A * np.abs(diff) + B)))
        user_item_diff.append({
            'userId': rating['userId'],
            'movieId': project_id,
            'rating': normalized_diff,
            'timestamp': rating['timestamp']
        })

    return user_item_diff


# 计算每个项目的历史平均分
df = pd.read_csv('../../datasets/preprocess_1M_0.75/train_set.csv', usecols=['movieId', 'rating'])
project_avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

# 逐行读取CSV文件并处理
chunk_size = 1000000  # 每次读取的行数
output_file = '../../datasets/preprocess_1M_0.5/1M_diff_0.5.csv'
results = []

with pd.read_csv('../../datasets/preprocess_1M_0.5/train_set.csv', chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        for user_id, user_ratings in chunk.groupby('userId'):
            diff_results = calculate_diff(user_ratings.to_dict(orient='records'), project_avg_ratings)
            results.extend(diff_results)

        # 保存结果到文件
        df_results = pd.DataFrame(results)
        if i == 0:
            df_results.to_csv(output_file, mode='w', header=['userId', 'movieId', 'rating', 'timestamp'], index=False)
        else:
            df_results.to_csv(output_file, mode='a', header=False, index=False)

        results.clear()  # 清空results以释放内存

print("处理完成，并保存到文件中。")
