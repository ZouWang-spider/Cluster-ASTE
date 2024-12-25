import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
def Sentiment_loss(pairs, final_outputs, triplet_labels):
    triplet_results = []
    total_loss = 0
    valid_pairs_count = 0  # 计数有效配对的数量

    # 输出final_outputs的shape以确认
    for idx, output in enumerate(final_outputs):
        # print(f"Final Output {idx} Shape: {output.shape}")  # torch.Size([3, 3])

        # 获取每个维度中的最大值
        _, predicted_classes = torch.max(output, dim=-1)
        # 预测的情感标签
        predicted_classes_list = predicted_classes.tolist()  # 转换为列表
        # 统计每个情感极性出现的频率
        counter = Counter(predicted_classes_list)
        # 选取出现次数最多的情感极性标签
        predict_label = counter.most_common(1)[0][0]  # 获取出现最多的情感标签

        # 获取对应的aspect-opinion配对
        aspect, opinion = pairs[idx]
        # 将结果组成三元组 (aspect, opinion, sentiment)
        triplet_results.append((aspect, opinion, predict_label))

        # 查找当前配对在triplet_labels中对应的标签（情感极性）
        for triplet in triplet_labels:
            # triplet = ([aspect_terms], [opinion_terms], sentiment)
            aspect_terms, opinion_terms, sentiment_label = triplet
            # if (set(aspect) == set(aspect_terms)) or (set(opinion) == set(opinion_terms)):
            if (set(aspect) == set(aspect_terms) or set(opinion) == set(opinion_terms)) or (set(aspect) == set(opinion_terms) or set(opinion) == set(aspect_terms)):
                # 真实情感标签，正确配对时才使用
                true_sentiment = torch.tensor([sentiment_label])  # 转换为tensor，表示真实情感极性

                # 如果配对正确，计算损失
                if sentiment_label != -1:
                    # CrossEntropyLoss 直接使用原始 logits，不需要再调用 softmax
                    loss = loss_fn(output[0].unsqueeze(0), true_sentiment)
                    total_loss += loss
                    valid_pairs_count += 1
                break
            else:
                # 对于配对错误，将情感极性标签赋值为-1，不计算损失
                continue

    # 计算平均损失
    # average_loss = total_loss / valid_pairs_count if valid_pairs_count > 0 else 0
    # 当没有有效的配对时，使用一个很小的常数，确保backward能够进行
    if valid_pairs_count > 0:
        average_loss = total_loss / valid_pairs_count
    else:
        average_loss = torch.tensor(1, dtype=torch.float, requires_grad=True)  # 使用一个很小的值

    return triplet_results, average_loss


###########################test#################################
# from Cluster_ASTE.BaseModel.Cross_Transformer import Cross_Transformer
# from Cluster_ASTE.ASTEModule.Aspect_opinion_pair import Cross_computing
# sentence = 'it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous .'
# Cross_model = Cross_Transformer(768)
# pairs, final_outputs = Cross_computing(sentence, Cross_model)
# triplet_labels = [([11], [15, 16], 1)]
#
# triplet_results, average_loss = Sentiment_loss(pairs, final_outputs, triplet_labels)
# print(triplet_results)
# print(average_loss)

