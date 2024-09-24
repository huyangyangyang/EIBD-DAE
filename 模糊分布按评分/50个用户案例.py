import numpy as np

# 示例用户的评分历史，格式为（用户ID，项目ID，评分，时间戳）
user_ratings = [
    (1, 101, 3, 100),
    (1, 102, 4, 200),
    (1, 103, 2, 300),
    (1, 104, 5, 400),
    (1, 105, 4, 500),
    (1, 106, 3, 600),
    (1, 107, 2, 700),
    (1, 108, 5, 800),
    (1, 109, 4, 900),
    (1, 110, 3, 1000),
    (1, 111, 3, 1100),
    (1, 112, 4, 1200),
    (1, 113, 2, 1300),
    (1, 114, 5, 1400),
    (1, 115, 4, 1500),
    (1, 116, 3, 1600),
    (1, 117, 2, 1700),
    (1, 118, 5, 1800),
    (1, 119, 4, 1900),
    (1, 120, 3, 2000),
    (1, 121, 3, 2100),
    (1, 122, 4, 2200),
    (1, 123, 2, 2300),
    (1, 124, 5, 2400),
    (1, 125, 4, 2500),
    (1, 126, 3, 2600),
    (1, 127, 2, 2700),
    (1, 128, 5, 2800),
    (1, 129, 4, 2900),
    (1, 130, 3, 3000),
    (1, 131, 3, 3100),
    (1, 132, 4, 3200),
    (1, 133, 2, 3300),
    (1, 134, 5, 3400),
    (1, 135, 4, 3500),
    (1, 136, 3, 3600),
    (1, 137, 2, 3700),
    (1, 138, 5, 3800),
    (1, 139, 4, 3900),
    (1, 140, 3, 4000),
    (1, 141, 3, 4100),
    (1, 142, 4, 4200),
    (1, 143, 2, 4300),
    (1, 144, 5, 4400),
    (1, 145, 4, 4500),
    (1, 146, 3, 4600),
    (1, 147, 2, 4700),
    (1, 148, 5, 4800),
    (1, 149, 4, 4900),
    (1, 150, 3, 5000)
]

# 分段阶段数
num_segments = 5

# 计算阶段时间点
timestamps = [rating[3] for rating in user_ratings]
segment_points = np.linspace(min(timestamps), max(timestamps), num_segments + 1)

# 计算每个阶段的起始时间和结束时间
segments = [(segment_points[i], segment_points[i+1]) for i in range(num_segments)]

# 计算每个阶段内的评分情况
ratings_per_segment = [[] for _ in range(num_segments)]
for rating in user_ratings:
    for i, segment in enumerate(segments):
        if segment[0] <= rating[3] <= segment[1]:
            ratings_per_segment[i].append(rating[2])

# 假设每个项目的平均评分
item_avg_ratings = {
    101: 3.5,
    102: 4.2,
    103: 5,
    104: 4.2,
    105: 2,
    106: 4,
    107: 3,
    108: 5,
    109: 2,
    110: 5,
    # 添加其他项目的平均评分...
}

# 计算每个阶段的加权平均值
segment_weighted_avg_ratings = {}
for i, segment in enumerate(segments):
    total_weighted_sum = 0
    total_membership = 0
    for rating in user_ratings:
        if segment[0] <= rating[3] <= segment[1]:
            membership_degree = (rating[3] - segment[0]) / (segment[1] - segment[0])  # 假设隶属度线性增加
            item_avg_rating = item_avg_ratings.get(rating[1], 0)  # 获取项目的平均评分，如果没有，则默认为0
            total_weighted_sum += item_avg_rating * membership_degree
            total_membership += membership_degree
    if total_membership != 0:
        segment_weighted_avg = total_weighted_sum / total_membership
    else:
        segment_weighted_avg = 0
    segment_weighted_avg_ratings[i] = segment_weighted_avg

# 计算每个用户对每个项目的评分与该项目隶属段的加权平均值之间的差异
user_item_diff = {}
for user_id, project_id, rating, timestamp in user_ratings:
    for i, segment in enumerate(segments):
        if segment[0] <= timestamp <= segment[1]:
            diff = rating - segment_weighted_avg_ratings[i]
            normalized_diff = 5 * (1 / (1 + np.exp(-1 * np.abs(diff) + 1)))  # 使用归一化函数对差异进行归一化处理
            user_item_diff[(user_id, project_id, i)] = normalized_diff

# 输出每个用户对每个项目的评分减去该项目隶属段的加权平均值的差异（归一化处理后）
for user_id, project_id, segment_index in user_item_diff.keys():
    diff = user_item_diff[(user_id, project_id, segment_index)]
    print(f'User {user_id} 对 Project {project_id} 在 Segment {segment_index + 1} 的归一化评分差异为 {diff}')
