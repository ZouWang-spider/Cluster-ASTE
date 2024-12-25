import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器F:\bert-base-cased
model_name = 'E:/bert-base-cased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 设置最大池化层
pool = nn.MaxPool1d(kernel_size=6, stride=6, padding=0)  #6
pool2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)  #6
def Word_embedding(text):
    marked_text1 = ["[CLS]"] + text + ["[SEP]"]
    # 将分词转化为词向量
    input_ids = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
    outputs = model(input_ids)
    # 获取词向量
    word_embeddings = outputs.last_hidden_state

    # 提取单词对应的词向量（去掉特殊标记的部分）
    word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
    # 提取单词（去掉特殊标记的部分）
    word_list = [item for item in marked_text1 if item not in ['[CLS]', '[SEP]']]
    # 使用切片操作去除第一个和最后一个元素
    word_feature = word_embeddings[0][1:-1, :]  # 节点特征

    # 应用最大池化
    pooled_embeddings = pool(word_feature)  #(n, 128)

    return word_list, word_feature, pooled_embeddings

