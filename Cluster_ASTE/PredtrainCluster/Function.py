def merge_mult_word_terms(text, cluster_labels, triplet_labels):
    # 如果 cluster_labels 的长度与 text 不一致，填充 cluster_labels 列表
    if len(cluster_labels) != len(text):
        cluster_labels.extend([2] * (len(text) - len(cluster_labels)))  # 填充标签2

    # 初始化合并后的文本和标签
    merged_text = []
    merged_cluster_labels = []
    updated_triplet_labels = []

    # 临时变量，用于追踪当前文本位置
    i = 0  # 指针，用来遍历三元组标签
    current_idx = 0  # 用来追踪原始文本的位置

    # 遍历三元组标签
    for triplet in triplet_labels:
        aspect_indices, opinion_indices, sentiment = triplet

        # 合并方面词
        aspect_tokens = [text[idx] for idx in aspect_indices]
        merged_aspect = ' '.join(aspect_tokens)

        # 合并观点词
        opinion_tokens = [text[idx] for idx in opinion_indices]
        merged_opinion = ' '.join(opinion_tokens)



        # 先把所有单词和聚类标签添加到merged_text和merged_cluster_labels
        while current_idx < len(text):
            # 如果当前词是需要合并的方面词之一
            if current_idx in aspect_indices:
                merged_text.append(merged_aspect)
                merged_cluster_labels.append(cluster_labels[aspect_indices[0]])  # 保持原聚类标签
                current_idx = aspect_indices[-1] + 1  # 跳到合并后的下一个词
            # 如果当前词是需要合并的观点词之一
            elif current_idx in opinion_indices:
                merged_text.append(merged_opinion)
                merged_cluster_labels.append(cluster_labels[opinion_indices[0]])  # 保持原聚类标签
                current_idx = opinion_indices[-1] + 1  # 跳到合并后的下一个词
            else:
                # 如果是普通单词，则不合并，直接加到文本
                merged_text.append(text[current_idx])
                merged_cluster_labels.append(cluster_labels[current_idx])
                current_idx += 1

        # 更新三元组标签的索引
        updated_triplet_labels.append(([i], [i + 1], sentiment))
        i += 2  # 每次合并两个部分

    return merged_text, merged_cluster_labels, updated_triplet_labels


# # 示例数据
# text = ['wonderful', 'features', '<', '.']
# cluster_labels = [1, 0, 2]
# triplet_labels = [([1], [0], 1)]
#
# # 调用合并函数
# merged_text, merged_cluster_labels, updated_triplet_labels = merge_mult_word_terms(text, cluster_labels, triplet_labels)
#
# # 打印输出结果
# print("Merged Text:", merged_text)
# print("Merged Cluster Labels:", merged_cluster_labels)
# print("Updated Triplet Labels:", updated_triplet_labels)
