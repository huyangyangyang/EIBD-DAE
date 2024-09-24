
import pandas as pd

# 读取csv文件
df = pd.read_csv('../datasets/preprocess_100k/test_set_filtered.csv')

# 将第三列名为“rating”的列都乘以5
df['rating'] = df['rating'] / 5

# 保存修改后的csv文件
df.to_csv('../datasets/preprocess_100k/test_set_filtered.csv', index=False)
