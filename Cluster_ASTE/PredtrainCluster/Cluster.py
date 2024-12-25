import os
import torch
import nltk
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from Cluster_ASTE.DataProcess.Process import Dataset_Process
from Cluster_ASTE.BaseModel.Embedding import Word_embedding
from Cluster_ASTE.BaseModel.Cluster_algorithm import MLP, SupConLoss

from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from Cluster_ASTE.PredtrainCluster.Function import merge_mult_word_terms


#名词短语检测
# import spacy
# nlp = spacy.load("en_core_web_sm")
# sentence = "The smartphone screen is bright, and the battery life is long."
# doc = nlp(sentence)
# for np in doc.noun_chunks:
#     print(np.text)

from scipy.optimize import linear_sum_assignment
def calculate_accuracy(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # 计算真实标签与预测标签之间的匹配矩阵
    D = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_labels))))

    for i in range(len(true_labels)):
        D[true_labels[i], predicted_labels[i]] += 1

    # 使用匈牙利算法找到最优标签匹配
    row_ind, col_ind = linear_sum_assignment(D.max() - D)

    # 计算准确率
    acc = sum([D[row, col] for row, col in zip(row_ind, col_ind)]) / float(len(true_labels))

    return acc


# 存储所有单词及对应的词向量
all_word_list = []
all_word_embeddings = []
all_labels = []

# 数据路径
file_path = r"E:\PythonProject2\Cluster_ASTE\triplet_datav2\16res\train_triplets.txt"

# 加载停用词列表
stop_words_path = r"E:\PythonProject2\ClusterASTE\DatasetCluster\stop_words.txt"
with open(stop_words_path, 'r') as f:
    stop_words = set(f.read().splitlines())

#cluster_labels:0为方面词、1为无关单词、2为观点词
datasets = Dataset_Process(file_path)
for sentence, cluster_labels, triplet_labels in datasets:
    print(sentence)
    text = nltk.word_tokenize(sentence)

    # 调用合并函数
    # merged_text, merged_cluster_labels, updated_triplet_labels = merge_mult_word_terms(text, cluster_labels,triplet_labels)

    pos_tag = nltk.pos_tag(text)
    pos_tags = [tag for word, tag in pos_tag]

    #单词嵌入
    word_list, word_feature, word_embeddings = Word_embedding(text)
    #词性嵌入
    pos_list, pos_features, pos_embeddings = Word_embedding(pos_tags)

    # 拼接两个张量
    feature_embeddings = torch.cat((word_feature, pos_embeddings), dim=1)  #768+256
    # print(feature_embeddings.shape)

    # 去除停用词
    filtered_word_list = []
    filtered_word_embeddings = []
    filtered_labels = []
    for word, embedding, label in zip(word_list, feature_embeddings, cluster_labels):
        if word.lower() not in stop_words:  # 检查词是否是停用词
            filtered_word_list.append(word)
            filtered_word_embeddings.append(embedding.detach().numpy())
            filtered_labels.append(label)  # 记录标签

        # 将处理后的词和词向量添加到全局列表
    all_word_list.extend(filtered_word_list)
    all_word_embeddings.append(np.array(filtered_word_embeddings))
    all_labels.extend(filtered_labels)  # 存储每个词对应的标签

# print(all_word_list)
# print(all_labels)


# 将所有词向量合并成一个矩阵
all_word_embeddings = np.vstack(all_word_embeddings)
all_labels = np.array(all_labels)  # 标签 (n_samples,)

# 将数据转化为 PyTorch 张量
word_embeddings_tensor = torch.tensor(all_word_embeddings, dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)

# 将数据包装成 DataLoader
dataset = TensorDataset(word_embeddings_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#使用MLP+CL有监督的聚类算法来训练聚类
input_dim = word_embeddings_tensor.shape[1]  # 输入维度（词向量维度）
embedding_dim = 128  # 嵌入维度
num_epochs = 50


mlp_model = MLP(input_dim=input_dim, embedding_dim=embedding_dim, drop_out=0.2)
criterion = SupConLoss(temperature=0.07)
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-4)


for epoch in range(num_epochs):
    mlp_model.train()  # 设置为训练模式
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空梯度

        # 通过 MLP 获取嵌入
        # inputs = inputs.unsqueeze(1)  # 调整为 (n, 1, 256)
        embeddings = mlp_model(inputs)

        # 计算对比损失
        loss = criterion(embeddings, labels)
        loss.backward()  # 反向传播

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 将模型设置为评估模式
    mlp_model.eval()

    # 存储所有样本的嵌入和标签
    embeddings_list = []
    labels_list = []

    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in dataloader:
            embeddings = mlp_model(inputs)  # 获取嵌入
            embeddings_list.append(embeddings.numpy())  # 转为numpy数组并添加到列表
            labels_list.append(labels.numpy())  # 记录对应的标签

    # 将嵌入转为numpy数组
    embeddings = np.vstack(embeddings_list)
    labels = np.hstack(labels_list)

    # 使用 K-Means 聚类
    num_clusters = 3  # 设置你想要的簇的数量
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)  # 聚类并返回标签

    # 计算 Accuracy
    acc_score = calculate_accuracy(labels, kmeans_labels)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {acc_score:.4f}')

    # 聚类评估：轮廓系数，范围是 [-1, 1]
    sil_score = silhouette_score(embeddings, kmeans_labels)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Silhouette Score: {sil_score:.4f}')

    # 计算 NMI (Normalized Mutual Information)，范围是 [0, 1]
    nmi_score = normalized_mutual_info_score(labels, kmeans_labels)
    print(f'Epoch [{epoch + 1}/{num_epochs}], NMI Score: {nmi_score:.4f}')

    # 计算 ARI (Adjusted Rand Index)，范围是 [-1, 1]
    ari_score = adjusted_rand_score(labels, kmeans_labels)
    print(f'Epoch [{epoch + 1}/{num_epochs}], ARI Score: {ari_score:.4f}')


# 保存MLP模型的参数
model_save_path = r'E:\PythonProject2\Cluster_ASTE\Model_path'
os.makedirs(model_save_path, exist_ok=True)
torch.save(mlp_model.state_dict(), os.path.join(model_save_path, 'mlp_model.pth'))

# 将模型设置为评估模式
mlp_model.eval()

# 存储所有样本的嵌入
embeddings_list = []
labels_list = []

with torch.no_grad():  # 不需要计算梯度
    for inputs, labels in dataloader:
        embeddings = mlp_model(inputs)  # 获取嵌入
        embeddings_list.append(embeddings.numpy())  # 转为numpy数组并添加到列表
        labels_list.append(labels.numpy())  # 记录对应的标签

# 将嵌入转为numpy数组
embeddings = np.vstack(embeddings_list)

# 使用 K-Means 聚类
num_clusters = 3  # 设置你想要的簇的数量
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(embeddings)  # 聚类并返回标签



###########################test#################################
# #聚类指标：轮廓系数，范围是 [-1, 1]月接近1说明聚类效果越好
# from sklearn.metrics import silhouette_score
# sil_score = silhouette_score(embeddings, kmeans_labels)
# print(f'Silhouette Score: {sil_score}')
#
# # NMI 衡量的是两个分布，范围[0, 1]
# from sklearn.metrics import normalized_mutual_info_score
# nmi_score = normalized_mutual_info_score(y_true, y_pred)
# print(f"NMI Score: {nmi_score}")
#
# #ARI是一个衡量聚类结果与真实标签之间相似度的指标，范围是 [-1, 1]
# from sklearn.metrics import adjusted_rand_score
# ari_score = adjusted_rand_score(y_true, y_pred)
# print(f"ARI Score: {ari_score}")


# 使用 t-SNE 将嵌入降维到 2D
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(reduced_embeddings[kmeans_labels == i, 0],
                reduced_embeddings[kmeans_labels == i, 1],
                label=f'Cluster {i+1}', s=30)
plt.title('K-Means Clustering with t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()





# from sklearn.metrics import silhouette_score
# sil_score = silhouette_score(all_word_embeddings, kmeans_labels)
# print(f'Silhouette Score: {sil_score}')  #0.045520685613155365  #0.17498809099197388  0.17635901272296906


# # 使用KMeans进行聚类
# kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
# kmeans.fit(all_word_embeddings)
#
# # 聚类标签
# labels = kmeans.labels_
#
# # 输出每个聚类的单词
# for cluster_id in range(kmeans.n_clusters):
#     cluster_words = [all_word_list[i] for i in range(len(labels)) if labels[i] == cluster_id]
#     print(f"Cluster {cluster_id}: {cluster_words[:50]}...")


# #可视化
# tsne = TSNE(n_components=2, random_state=42)
# word_embeddings_2d = tsne.fit_transform(all_word_embeddings)
#
# # 绘制散点图
# plt.figure(figsize=(10, 10))
# scatter = plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
# plt.colorbar(scatter)
# plt.title('t-SNE visualization of clustered word embeddings')
# plt.show()
#
# from sklearn.metrics import silhouette_score
# sil_score = silhouette_score(all_word_embeddings, labels)
# print(f'Silhouette Score: {sil_score}')  #0.045520685613155365  #0.17498809099197388  0.17635901272296906



# from sklearn.cluster import AgglomerativeClustering
#
# # 使用层次聚类
# agg_clust = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
#
# # 进行聚类
# labels = agg_clust.fit_predict(all_word_embeddings)
#
# # 输出每个聚类的单词
# for cluster_id in range(3):
#     cluster_words = [all_word_list[i] for i in range(len(labels)) if labels[i] == cluster_id]
#     print(f"Cluster {cluster_id}: {cluster_words[:50]}...")
#
#
# from sklearn.metrics import silhouette_score
# sil_score = silhouette_score(all_word_embeddings, labels)
# print(f'Silhouette Score: {sil_score}')
#
#
# import scipy.cluster.hierarchy as sch
# import matplotlib.pyplot as plt
#
# # 计算层次聚类的链接矩阵
# Z = sch.linkage(all_word_embeddings, method='ward', metric='euclidean')
#
# # 绘制树状图
# plt.figure(figsize=(10, 7))
# sch.dendrogram(Z)
# plt.title('Dendrogram for Word Embeddings')
# plt.xlabel('Words')
# plt.ylabel('Distance')
# plt.show()