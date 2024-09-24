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
        if len(user_item_diff) == 10:  # 检查前10行
            return user_item_diff

    return user_item_diff


# 定义要等分的段数
m_segments = 10

# 计算每个项目的历史平均分（需要提前读取整个文件的此部分）
df = pd.read_csv('../../datasets/goodreads/train_set.csv')
project_avg_ratings = df.groupby('movieId')['rating'].mean().to_dict()

# 按行读取CSV文件并处理
results = []
with pd.read_csv('../../datasets/goodreads/train_set.csv', chunksize=1) as reader:
    user_ratings = []
    current_user = None
    for chunk in reader:
        user_id = chunk['userId'].iloc[0]
        if current_user is None:
            current_user = user_id

        # 如果是新用户，先处理上一个用户的数据
        if user_id != current_user:
            results.extend(calculate_diff(user_ratings, project_avg_ratings))
            current_user = user_id
            user_ratings = []

        user_ratings.append(chunk.iloc[0].to_dict())

        # 检查结果是否已达到10行
        if len(results) >= 10:
            break

# 转换结果为DataFrame并输出前10行
diff_results_df = pd.DataFrame(results, columns=['userId', 'movieId', 'rating', 'timestamp'])
print(diff_results_df.head(10))

# 继续处理其他数据并保存到文件
# with open('output_file.csv', 'a') as f_out:
#     diff_results_df.to_csv(f_out, header=f_out.tell()==0, index=False)
