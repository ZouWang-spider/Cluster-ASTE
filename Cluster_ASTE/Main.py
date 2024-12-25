import torch
import torch.optim as optim


from Cluster_ASTE.DataProcess.Process import Dataset_Process
from Cluster_ASTE.BaseModel.Cross_Transformer import Cross_Transformer
from Cluster_ASTE.ASTEModule.Aspect_opinion_pair import Cross_computing
from Cluster_ASTE.Loss_computing import Sentiment_loss

from Cluster_ASTE.F1_score_computing import calculate_f1


#初始化Cross Transformer
hidden_dim = 768
Cross_model = Cross_Transformer(hidden_dim)


# 数据路径
file_path = r"E:\PythonProject2\Cluster_ASTE\triplet_datav2\14lap\test_triplets.txt"

# 优化器
optimizer = optim.Adam(list(Cross_model.parameters()) ,lr=1e-4)

# 训练轮次
num_epochs = 50

if __name__ == "__main__":

    datasets = Dataset_Process(file_path)

    # 开始训练
    for epoch in range(num_epochs):
        total_epoch_loss = 0  # 每轮训练的总损失
        all_triplet_results = []  # 存储所有 batch 的预测结果
        all_triplet_labels = []  # 存储所有 batch 的真实标签
        batch_count = 0  # 统计batch数量

        for sentence, cluster_labels, triplet_labels in datasets:

            batch_count += 1  # 累加batch数量
            print(sentence)

            #计算三元组情感损失
            pairs, final_outputs = Cross_computing(sentence, Cross_model)
            print(pairs)
            print(triplet_labels)


            #计算配对情感的损失
            triplet_results, sentiment_loss = Sentiment_loss(pairs, final_outputs, triplet_labels)
            print(sentiment_loss)

            # 将batch的预测结果和标签保存
            all_triplet_results.extend(triplet_results)
            all_triplet_labels.extend(triplet_labels)

            #总损失计算并优化
            optimizer.zero_grad()
            sentiment_loss.backward()
            optimizer.step()

            total_epoch_loss += sentiment_loss.item()
            print('OK')

        #计算每轮的平均损失
        average_epoch_loss = total_epoch_loss / batch_count

        #计算Sentiment的 Precision, Recall, F1 Score
        precision, recall, f1_score, correct_triplets = calculate_f1(all_triplet_results, all_triplet_labels)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        # 打开文件，写入训练结果
        with open('E:/PythonProject2/Cluster_ASTE/Train_epoch/epoch.txt', 'a') as f:
            # 在每个 Epoch 结束时记录损失和准确率
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")

    print("Training completed!")