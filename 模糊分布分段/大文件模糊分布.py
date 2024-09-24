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
    weighted_avg_ratings = np.zeros(num_segments)
    counts = np.zeros(num_segments)

    for rating in user_ratings:
        timestamp = rating['timestamp']
        segment_index = (timestamp // 100) % num_segments  # 修正分段计算
        membership_degree = trapezoidal_mf(timestamp, segment_index * 100, segment_index * 100 + 20,
                                           segment_index * 100 + 80, segment_index * 100 + 100)
        item_avg_rating = project_avg_ratings.get(rating['movieId'], 0)

        weighted_avg_ratings[segment_index] += item_avg_rating * membership_degree
        counts[segment_index] += membership_degree

    # 计算每个分段的加权平均评分
    weighted_avg_ratings /= np.where(counts != 0, counts, 1)

    user_item_diff = []
    for rating in user_ratings:
        project_id = rating['movieId']
        segment_index = (rating['timestamp'] // 100) % num_segments  # 修正分段计算
        weighted_avg = weighted_avg_ratings[segment_index]
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
df = pd.read_csv('../../datasets/goodreads_0.5/train_set.csv', usecols=['movieId', 'rating'])
project_avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

# 逐块读取CSV文件并处理
chunk_size = 1000000  # 每次读取的行数
output_file = '../../datasets/goodreads_0.5/goodreads_diff_0.5.csv'

with pd.read_csv('../../datasets/goodreads_0.5/train_set.csv', chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        results = []
        for user_id, user_ratings in chunk.groupby('userId'):
            diff_results = calculate_diff(user_ratings.to_dict(orient='records'), project_avg_ratings)
            results.extend(diff_results)

        # 保存结果到文件
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, mode='a', header=(i == 0), index=False)

        print(f"处理完成: {i + 1} 块数据")

print("所有数据处理完成，并保存到文件中。")
