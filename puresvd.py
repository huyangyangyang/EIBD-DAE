import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 数据预处理
data = pd.read_csv('datasets/preprocess_1M_0.5/1M_0.5.csv')

# 用户和电影ID映射到整数
user_ids = data['userId'].astype('category').cat.codes.values
movie_ids = data['movieId'].astype('category').cat.codes.values
ratings = data['rating'].values

# 划分训练集和测试集
train_data, test_data = train_test_split(pd.DataFrame({'user': user_ids, 'item': movie_ids, 'rating': ratings}),
                                         test_size=0.2, random_state=42)

# 转换为PyTorch数据集
def create_tensor_dataset(df):
    users = torch.tensor(df['user'].values, dtype=torch.long)
    items = torch.tensor(df['item'].values, dtype=torch.long)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    return TensorDataset(users, items, ratings)

train_dataset = create_tensor_dataset(train_data)
test_dataset = create_tensor_dataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

# 定义PureSVD模型
class PureSVD(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(PureSVD, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.svd_weight = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.loss_fn = nn.MSELoss()

    def forward(self, user, item):
        user_embeds = self.user_embedding(user)
        item_embeds = self.item_embedding(item)
        svd = torch.matmul(user_embeds, self.svd_weight)
        return torch.sum(svd * item_embeds, dim=1)

    def fit(self, train_loader, epochs, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for user, item, rating in train_loader:
                optimizer.zero_grad()
                output = self(user, item)
                loss = self.loss_fn(output, rating)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

# 训练模型
num_users = len(data['userId'].unique())
num_items = len(data['movieId'].unique())
latent_dim = 10
epochs = 500
lr = 1e-3

model = PureSVD(num_users, num_items, latent_dim)
model.fit(train_loader, epochs, lr)

# 评估模型
def evaluate_model(model, test_loader, top_k):
    model.eval()
    all_users = []
    all_items = []
    all_ratings = []
    predictions = []

    with torch.no_grad():
        for user, item, rating in test_loader:
            output = model(user, item)
            predictions.extend(output.tolist())
            all_users.extend(user.tolist())
            all_items.extend(item.tolist())
            all_ratings.extend(rating.tolist())

    # 创建用户-项目评分的预测字典
    user_item_scores = defaultdict(list)
    for user, item, score in zip(all_users, all_items, predictions):
        user_item_scores[user].append((item, score))

    # 计算 Precision@K, Recall@K 和 NDCG@K
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []

    for user in user_item_scores:
        # 排序并选择Top-K推荐
        user_scores = sorted(user_item_scores[user], key=lambda x: x[1], reverse=True)
        top_k_items = set([item for item, score in user_scores[:top_k]])

        # 真实评分
        true_ratings = set(test_data[(test_data['user'] == user) & (test_data['rating'] >= 4)]['item'])

        # 计算 Precision@K 和 Recall@K
        relevant_items = len(true_ratings.intersection(top_k_items))
        precision_at_k.append(relevant_items / top_k)
        recall_at_k.append(relevant_items / len(true_ratings) if len(true_ratings) > 0 else 0)

        # 计算 NDCG@K
        relevance = [1 if item in true_ratings else 0 for item, _ in user_scores[:top_k]]
        ndcg_at_k.append(ndcg_score([relevance], [list(range(len(relevance)))]))

    print(f'Precision@{top_k}: {np.mean(precision_at_k)}')
    print(f'Recall@{top_k}: {np.mean(recall_at_k)}')
    print(f'NDCG@{top_k}: {np.mean(ndcg_at_k)}')

# 设置Top-K值并评估模型
top_k = 5  # 根据需要设置Top-K值
evaluate_model(model, test_loader, top_k)
