import torch
import numpy as np
import nltk
from collections import Counter
from sklearn.cluster import KMeans


from Cluster_ASTE.BaseModel.Cluster_algorithm import MLP
from Cluster_ASTE.BaseModel.Embedding import Word_embedding
from Cluster_ASTE.BaseModel.BiAffine import Dependency_relation

from Cluster_ASTE.BaseModel.Boundary_aware import Multword_term_aware, Extract_phrases_from_indices

#Discrimination using clustering
def pretrained_clustering(sentence):
    text = nltk.word_tokenize(sentence)
    word_list, word_feature, word_embeddings = Word_embedding(text)

    # word embedding
    pos_tag = nltk.pos_tag(text)
    pos_tags = [tag for word, tag in pos_tag]
    pos_list, word_features, pos_embeddings = Word_embedding(pos_tags)

    # concat two embedding
    feature_embeddings = torch.cat((word_feature, pos_embeddings), dim=1)  #768+128

    # stop word
    stop_words_path = r"E:\PythonProject2\Cluster_ASTE\PredtrainCluster\stop_words.txt"
    with open(stop_words_path, 'r') as f:
        stop_words = set(f.read().splitlines())

    # remove stop word
    filtered_word_list = []
    filtered_feature_embeddings = []
    for word, embedding in zip(word_list, feature_embeddings):
        if word.lower() not in stop_words:  # check if in stop word list
            filtered_word_list.append(word)
            filtered_feature_embeddings.append(embedding)

    # stack tensor
    filtered_feature_embeddings = torch.stack(filtered_feature_embeddings)

    # dim word embedding
    input_dim = filtered_feature_embeddings.shape[1]
    embedding_dim = 128  # dim

    # load per-train clusterer
    mlp_model = MLP(input_dim=input_dim, embedding_dim=embedding_dim)
    mlp_model.load_state_dict(torch.load(r'E:\PythonProject2\Cluster_ASTE\Model_path\14lap_mlp_model2.pth'))
    mlp_model.eval()  # model eval

    # get feature embedding
    with torch.no_grad():
        embeddings = mlp_model(filtered_feature_embeddings.float())

    # using KMeans clustering
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings.numpy())  # return labels {aspect cluster, opinion cluster, irrelevant cluster}

    # Get the center of each cluster
    cluster_centers = kmeans.cluster_centers_

    # Sorting the centers
    sorted_labels = np.argsort(cluster_centers[:, 0])

    # Mapping of tags to ensure tag consistency (0: irrelevant words, 1: viewpoint words, 2: aspect words)
    sorted_kmeans_labels = np.array([sorted_labels[label] for label in kmeans_labels])

    return filtered_word_list, sorted_kmeans_labels


# # Get aspect and opinion index list
# def Get_aspect_opinion_index(sentence):
#     #Calling pre-trained clusterer to achieve discrimination
#     filtered_word_list, kmeans_labels = pretrained_clustering(sentence)
#
#     # change list
#     kmeans_labels = kmeans_labels.tolist()
#     print(kmeans_labels)
#
#     # Initialize aspect set and opinion set
#     aspect_sets = []
#     opinion_sets = []
#     # Iterate over word lists and clustering labels for classification
#     for i, (word, label) in enumerate(zip(filtered_word_list, kmeans_labels)):
#         if label == 1:  # aspect
#             aspect_sets.append(word)
#         elif label == 2:  # opinion
#             opinion_sets.append(word)
#
#     # word tokenize
#     word_list = nltk.word_tokenize(sentence)
#     # Get aspect index and opinion index
#     aspect_index = [i for i, word in enumerate(word_list) if word in aspect_sets]
#     opinion_index = [i for i, word in enumerate(word_list) if word in opinion_sets]
#
#     return aspect_index, opinion_index

# Get aspect index and opinion index
def Get_aspect_opinion_index(sentence):
    # Calling pre-trained clusterer discrimination
    filtered_word_list, kmeans_labels = pretrained_clustering(sentence)

    # change list
    kmeans_labels = kmeans_labels.tolist()
    # print('predict_label', kmeans_labels)

    # Counting the number of each label
    label_counts = Counter(kmeans_labels)

    # Get the two tags with the least number of tag occurrences
    min_labels = label_counts.most_common()

    # Determine if tags are balanced (if there are multiple tags and equal number of tags)
    if len(min_labels) > 2 and min_labels[0][1] != min_labels[1][1]:
        # If there are two tags with the lowest number of tags, select them
        min_labels = [label for label, count in min_labels[-2:]]  # Get the two tags with the lowest number of tags
    else:
        # If the number of tags is equal, use the default method to extract aspect and opinion
        min_labels = [2, 1]  # 2 for aspect and 1 for opinion

    # Initialize aspect set and opinion set
    aspect_sets = []
    opinion_sets = []

    # Categorization of selected tags
    for i, (word, label) in enumerate(zip(filtered_word_list, kmeans_labels)):
        if label == min_labels[0]:  # first label is aspect
            aspect_sets.append(word)
        elif label == min_labels[1]:  # second label is opinion
            opinion_sets.append(word)

    # word tokenize
    word_list = nltk.word_tokenize(sentence)

    # get aspect index and opinion index
    aspect_index = [i for i, word in enumerate(word_list) if word in aspect_sets]
    opinion_index = [i for i, word in enumerate(word_list) if word in opinion_sets]

    return aspect_index, opinion_index


#Use dependencies for ATE and OTE
def multword_Awareness(sentence):
    #get sentence graph and dependencies
    relation, graph = Dependency_relation(sentence)

    #get aspect index and opinion index
    aspect_indices, opinion_indices = Get_aspect_opinion_index(sentence)

    #boundary awareness for multi-word
    Aware_aspect_indices, Aware_opinion_indices = Multword_term_aware(relation, graph, aspect_indices, opinion_indices)

    return Aware_aspect_indices, Aware_opinion_indices


def ATE_OTE(sentence):

    #get aspect index and opinion index
    Aware_aspect_indices, Aware_opinion_indices = multword_Awareness(sentence)

    #ATE and  OTE
    aspect_phrases, opinion_phrases = Extract_phrases_from_indices(sentence, Aware_aspect_indices, Aware_opinion_indices)


    return aspect_phrases, opinion_phrases



###########################Test#################################
# sentence = 'Enabling the battery timer is useless .'
#
# filtered_word_list, sorted_kmeans_labels = pretrained_clustering(sentence)
# # print(filtered_word_list)
# # print(sorted_kmeans_labels)
#
# aspect_index, opinion_index = Get_aspect_opinion_index(sentence)
# print(aspect_index)
# print(opinion_index)

# Aware_aspect_indices, Aware_opinion_indices = multword_Awareness(sentence)
# print(Aware_aspect_indices)
# print(Aware_opinion_indices)
# aspect_phrases, opinion_phrases = ATE_OTE(sentence)
# print(aspect_phrases)
# print(opinion_phrases)


