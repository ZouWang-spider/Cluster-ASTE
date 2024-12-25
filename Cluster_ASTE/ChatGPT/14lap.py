import re
from sklearn.metrics import precision_score, recall_score, f1_score

import ast  # 用于将字符串转换为列表


def extract_sentence_and_labels(input_str):
    """
    提取句子和标签部分
    """
    # 按 '##' 分割句子和标签
    sentence, label_str = input_str.split('##')

    # 去除多余的空白字符
    sentence = sentence.strip()

    # 将标签部分从字符串转换为 Python 列表
    label_info = ast.literal_eval(label_str.strip())

    return sentence, label_info


def extract_triplets_from_positions(sentence, triplet_infos):

    # 将句子拆分成单词列表
    words = sentence.split()

    # 情感标签映射
    sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}

    # 用于存储提取的三元组
    triplets = []

    # 遍历每个三元组的位置信息
    for triplet_info in triplet_infos:
        aspect_positions, opinion_positions, sentiment_label = triplet_info

        # 提取方面词
        aspect = " ".join([words[i] for i in aspect_positions])

        # 提取观点词
        opinion = " ".join([words[i] for i in opinion_positions])

        # 获取情感标签
        sentiment = sentiment_map.get(sentiment_label, 'neutral')  # 默认值为 'neutral'

        # 添加三元组到列表中
        triplets.append((aspect, opinion, sentiment))

    return triplets



# 定义函数计算 Precision, Recall 和 F1
# 定义函数计算 Precision, Recall 和 F1
def calculate_metrics(true_triplets, predicted_triplets):
    """
    计算 Precision, Recall 和 F1 Score。
    """
    # 将真实标签和预测标签转换为字符串形式，便于计算
    true_labels = [f"{aspect} {opinion} {sentiment}" for aspect, opinion, sentiment in true_triplets]
    predicted_labels = [f"{aspect} {opinion} {sentiment}" for aspect, opinion, sentiment in predicted_triplets]

    # 补充缺失的预测标签
    if len(true_labels) > len(predicted_labels):
        missing_count = len(true_labels) - len(predicted_labels)
        predicted_labels.extend(["missing"] * missing_count)  # 填充缺失的预测标签

    # 使用精确匹配的方式计算 Precision, Recall, F1 Score
    precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)

    return precision, recall, f1



# 读取真实标签和预测标签的文件
true_triplets = []
predicted_triplets = []

with open('14lap_data.txt', 'r') as f_data, open('14lap_triplet.txt', 'r') as f_triplet:
    for data_line, triplet_line in zip(f_data, f_triplet):
        # 提取句子和三元组位置信息
        sentence, triplet_info = extract_sentence_and_labels(data_line.strip())
        true_triplet = extract_triplets_from_positions(sentence, triplet_info)

        true_triplets.extend(true_triplet)  # 添加到真实标签列表

        # 提取预测的三元组
        triplet_line = triplet_line.strip()  # 去掉换行符
        # 去掉括号并分割
        triplet_line = triplet_line.strip('()').replace(', ', ',')  # 去掉括号并确保没有多余的空格
        predicted_triplet = tuple(triplet_line.split(','))

        if len(predicted_triplet) == 3:  # 确保我们得到了三个元素
            aspect, opinion, sentiment = predicted_triplet

            predicted_triplets.append((aspect.strip(), opinion.strip(), sentiment.strip()))  # 添加到预测标签列表
        else:
            print(f"Invalid triplet format: {predicted_triplet}")


# 打印真实标签和预测标签以检查
print(f"True triplets: {true_triplets}")
print(f"Predicted triplets: {predicted_triplets}")

# 计算 Precision, Recall 和 F1
precision, recall, f1 = calculate_metrics(true_triplets, predicted_triplets)

# 打印结果
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
