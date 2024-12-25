import torch
import nltk
from sklearn.cluster import KMeans


from Cluster_ASTE.BaseModel.Cluster_algorithm import MLP, SupConLoss
from Cluster_ASTE.BaseModel.Embedding import Word_embedding

#I charge it at night and skip taking the cord with me because of the good battery life .
sentence = 'The place is small and terrible but the food is fantastic .'
text = nltk.word_tokenize(sentence)
word_list, word_feature, word_embeddings = Word_embedding(text)

#词性嵌入
pos_tag = nltk.pos_tag(text)
pos_tags = [tag for word, tag in pos_tag]
pos_list, pos_features, pos_embeddings = Word_embedding(pos_tags)

# 拼接两个张量
feature_embeddings = torch.cat((word_feature, pos_embeddings), dim=1)  #768+128

# 加载停用词列表
stop_words_path = r"E:/PythonProject2/Cluster_ASTE/PredtrainCluster/stop_words.txt"
with open(stop_words_path, 'r') as f:
    stop_words = set(f.read().splitlines())

# 去除停用词
filtered_word_list = []
filtered_feature_embeddings = []
for word, embedding in zip(word_list, feature_embeddings):
    if word.lower() not in stop_words:  # 检查词是否是停用词
        filtered_word_list.append(word)
        filtered_feature_embeddings.append(embedding)


# 将过滤后的嵌入转为 tensor
filtered_feature_embeddings = torch.stack(filtered_feature_embeddings)


# 输入维度（词向量维度）
input_dim = filtered_feature_embeddings.shape[1]
embedding_dim = 128  # 嵌入维度

# 加载模型
mlp_model = MLP(input_dim=input_dim, embedding_dim=embedding_dim)
mlp_model.load_state_dict(torch.load(r'E:\PythonProject2\Cluster_ASTE\Model_path\15res_mlp_model_896.pth'))
mlp_model.eval()  # 设置为评估模式

# 获取单词的嵌入
with torch.no_grad():
    embeddings = mlp_model(filtered_feature_embeddings.float())  # 获取经过 MLP 的嵌入

# 使用 KMeans 聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(embeddings.numpy())  # 聚类并返回标签

# 打印聚类结果
print("Words and their predicted cluster labels:")
for word, label in zip(filtered_word_list, kmeans_labels):
    print(f"{word}: Cluster {label}")





