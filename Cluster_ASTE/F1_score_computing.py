from sklearn.metrics import precision_score, recall_score, f1_score




#计算预测三元组的P、R、F1 score
def calculate_f1(triplet_results, triplet_labels):
    correct_triplets = []  # 存储配对正确的元组

    # 遍历预测的三元组
    for predicted_triplet in triplet_results:
        aspect_pred, opinion_pred, sentiment_pred = predicted_triplet

        # 遍历真实标签的三元组
        for true_triplet in triplet_labels:
            aspect_true, opinion_true, sentiment_true = true_triplet

            #聚类会出现方面词与观点词的标签互换等问题
            #(aspect,opinion)<==>(opinion,aspect)更符合聚类结果判断   and sentiment_pred == sentiment_true
            if (set(aspect_pred) == set(aspect_true)) or (set(aspect_pred) == set(opinion_true)):
            # if (set(aspect_pred) == set(aspect_true) and  set(opinion_pred) == set(opinion_true))  or (set(aspect_pred) == set(opinion_true) and set(opinion_pred) == set(aspect_true)) :
                correct_triplets.append(predicted_triplet)  # 添加到正确预测的列表中
                break  # 跳出内层循环，避免重复计数

    # 转换为集合（将嵌套 list 转为 tuple）
    predicted_set = set([tuple(map(tuple, t[:2])) + (t[2],) for t in triplet_results])
    true_set = set([tuple(map(tuple, t[:2])) + (t[2],) for t in triplet_labels])
    correct_set = set([tuple(map(tuple, t[:2])) + (t[2],) for t in correct_triplets])

    # 计算 Precision, Recall, F1 Score
    tp = len(correct_set)  # True Positives
    fp = len(predicted_set - correct_set)  # False Positives
    fn = len(true_set - correct_set)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score, correct_triplets


# def calculate_f1(triplet_results, triplet_labels):
#     # 转换为集合格式（将嵌套 list 转为 tuple）
#     predicted_set = set([tuple(map(tuple, t[:2])) + (t[2],) for t in triplet_results])
#     true_set = set([tuple(map(tuple, t[:2])) + (t[2],) for t in triplet_labels])
#
#     # 计算 True Positives (交集)
#     correct_set = predicted_set & true_set  # 交集表示正确预测
#
#     # 计算 Precision, Recall, F1 Score
#     tp = len(correct_set)  # True Positives
#     fp = len(predicted_set - correct_set)  # False Positives
#     fn = len(true_set - correct_set)  # False Negatives
#
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#
#     # 返回精确率、召回率、F1分数以及正确预测的三元组
#     return precision, recall, f1_score, list(correct_set)
