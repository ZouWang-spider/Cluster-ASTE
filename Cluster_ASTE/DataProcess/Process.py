import re
import ast


# extraction triplet
def extract_triplets(sentence, triplet_indices):
    tokens = sentence.split()
    triplets = []
    for aspect_indices, opinion_indices, sentiment in triplet_indices:
        aspect = " ".join(tokens[i] for i in aspect_indices)
        opinion = " ".join(tokens[i] for i in opinion_indices)
        triplets.append((aspect, opinion, sentiment))
    return triplets


def Dataset_Process(file_path):
    data = []
    # red dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # get Sentence
            parts = line.split("####")
            sentences = parts[0].strip()
            triplet_information = eval(parts[-1].strip())  # 使用 eval 将字符串解析为 Python 对象


            #获取聚类标签
            sentence_length = len(sentences.split())
            # 初始化所有节点的标签为 2
            cluster_labels = [2] * sentence_length
            # 遍历 final_info，标记方面词和观点词的标签
            for aspect_terms, opinion_terms, _ in triplet_information:
                # 设置方面词为 0
                for idx in aspect_terms:
                    if 0 <= idx < sentence_length:  # 确保索引有效
                        cluster_labels[idx] = 0
                # 设置观点词为 1
                for idx in opinion_terms:
                    if 0 <= idx < sentence_length:  # 确保索引有效
                        cluster_labels[idx] = 1
            # print(cluster_labels)


            #获取三元组标签
            sentiment_mapping = {'NEG': 0, 'POS': 1, 'NEU': 2}
            # 遍历并替换情感标签
            triplet_labels = [
                (aspect, opinion, sentiment_mapping[sentiment])
                for aspect, opinion, sentiment in triplet_information
            ]

            data.append((sentences, cluster_labels, triplet_labels))
    return data


###########################test#################################
# file_path = r"E:\PythonProject2\Cluster_ASTE\triplet_datav2\14lap\test_triplets.txt"
# datasets = Dataset_Process(file_path)
# # print(datasets)
# for sentence, cluster_labels, triplet_labels in datasets:
#     print(sentence)
#     # print(triplet_labels)

