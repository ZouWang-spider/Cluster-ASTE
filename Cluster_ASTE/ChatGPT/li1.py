def extract_triplets_from_positions(sentence, triplet_infos):
    """
    根据多个三元组的位置信息提取方面词、观点词和情感，返回完整的三元组列表。

    参数：
    sentence: 输入的句子
    triplet_infos: 多个三元组的位置信息列表，形式为 [([aspect_positions], [opinion_positions], sentiment), ... ]

    返回：
    完整的三元组列表 [(aspect, opinion, sentiment), ... ]
    """
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

# 示例输入
sentence = "It has so much more speed and the screen is very sharp ."
triplet_infos = [([5], [3, 4], 1), ([8], [11], 1)]

# 调用函数提取三元组
triplets = extract_triplets_from_positions(sentence, triplet_infos)

# 输出结果
print(triplets)
