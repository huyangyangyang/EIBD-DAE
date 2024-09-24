import numpy as np
import matplotlib.pyplot as plt

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
    (1, 110, 3, 1000)
]

# 根据评分时间排序
user_ratings.sort(key=lambda x: x[3])

# 分段阶段数
num_segments = 4

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

# 计算每个阶段内评分的梯形模糊分布
def trapezoidal_mf(x, start, peak_start, peak_end, end):
    if x <= start or x >= end:
        return 0
    elif peak_start <= x <= peak_end:
        return 1
    elif start < x < peak_start:
        return (x - start) / (peak_start - start)
    else:
        return (end - x) / (end - peak_end)

# 绘制每个阶段内的梯形模糊分布
plt.figure(figsize=(12, 6))
for i, segment in enumerate(segments):
    plt.subplot(1, num_segments, i+1)
    x_values = np.linspace(segment[0], segment[1], 100)
    y_values = [trapezoidal_mf(x, segment[0], segment[0] + (segment[1] - segment[0]) * 0.2, segment[0] + (segment[1] - segment[0]) * 0.8, segment[1]) for x in x_values]
    plt.plot(x_values, y_values, label=f'Segment {i+1}')
    plt.xlabel('Timestamp')
    plt.ylabel('Membership Degree')
    plt.title(f'Segment {i+1}')
    plt.grid(True)
plt.tight_layout()
plt.show()
