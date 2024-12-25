import torch
import torch.nn as nn
import torch.nn.functional as F

class Cross_Transformer(nn.Module):
    def __init__(self, hidden_dim):
        super(Cross_Transformer, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 假设这里定义了注意力层和MLP层的参数
        self.mlp_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # 保持原来的层
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # 线性层将256维的输出映射到类别数
        self.classifier = nn.Linear(256, 3)

        # # 概率分布
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # 将输入张量通过权重矩阵变换
        query_transformed = self.query_layer(query)
        key_transformed = self.key_layer(key)
        value_transformed = self.value_layer(value)

        # 注意力机制
        attn_weights = torch.matmul(query_transformed, key_transformed.transpose(0, 1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 使用注意力权重对 Value 进行加权求和
        output = torch.matmul(attn_weights, value_transformed)
        # print(output.shape) #（10,768）

        # Add & Layer Normalization
        attention_output = self.layer_norm(output + query)

        # MLP
        mlp_output = self.mlp_layer(attention_output)

        # 分类器将256维映射到类别数
        logits = self.classifier(mlp_output)

        # output = self.softmax(logits)  # 输出情感分类的概率

        return logits



###########################Test#################################
# aspect_opinion = torch.randn(5, 768)
# text = torch.randn(10, 768)  #Query
# #模型初始化
# Cross_model = Cross_Transformer(768)
# final_output = Cross_model(aspect_opinion, text, text)
# print(final_output)

