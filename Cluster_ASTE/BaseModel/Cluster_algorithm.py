import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# 定义一个简单的神经网络进行嵌入学习
class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, drop_out=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, embedding_dim)
        self.dropout = nn.Dropout(drop_out)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x




# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=128, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接层，将LSTM的输出映射到目标embedding维度
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # 初始化隐藏层状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取LSTM的最后一个时刻的输出（用于生成嵌入向量）
        out = out[:, -1, :]  # 只取最后一个时间步的输出

        # 将LSTM的输出映射到指定维度的embedding
        out = self.fc(out)
        return out

#对比损失函数
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]

        # 将标签调整为列向量
        labels = labels.contiguous().view(-1, 1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 对角线元素（即样本与自己之间的相似度）需要排除在外
        mask = torch.eq(labels, labels.T).float()

        # 在计算相似度时，确保不会将样本与自身进行比较
        mask = mask - torch.eye(batch_size).float() # 如果你用的是 GPU，使用 .cuda()

        # 计算正样本和负样本的相似度
        positive_sim = mask * similarity_matrix
        negative_sim = (1 - mask) * similarity_matrix

        # 为避免数值溢出，加入一个微小的常数 epsilon
        epsilon = 1e-8
        positive_sim = torch.clamp(positive_sim, min=-20.0)  # 保证数值范围
        negative_sim = torch.clamp(negative_sim, min=-20.0)

        # 计算对比损失
        positive_exp = torch.exp(positive_sim)
        negative_exp = torch.exp(negative_sim)

        # 计算损失：避免除零
        loss = -torch.log(positive_exp / (torch.sum(negative_exp, dim=1) + epsilon)).mean()

        return loss



###########################Test#################################
# features = torch.randn(100, 50)  # 100个样本，50维特征
# labels = torch.randint(0, 3, (100,))  # 假设3类标签
#
# # 训练模型
# model = MLP(256, 128)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = SupConLoss()
#
# for epoch in range(10):
#     model.train()
#     optimizer.zero_grad()
#     embeddings = model(features)
#     loss = loss_fn(embeddings, labels)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
#
# # 用K-means聚类
# embeddings = model(features).detach().numpy()
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(embeddings)
# cluster_labels = kmeans.labels_
#
# # 计算聚类质量
# ari = adjusted_rand_score(labels.numpy(), cluster_labels)
# print(f"Adjusted Rand Index: {ari}")
