import nltk
import torch
from nltk import Tree
from collections import Counter
from stanfordcorenlp import StanfordCoreNLP


from Cluster_ASTE.BaseModel.ConstituentTree import Get_paragraph_word_indices
from Cluster_ASTE.ASTEModule.ATE_OTE import multword_Awareness
from Cluster_ASTE.BaseModel.Pair_Process import group_consecutive_indices, pair_aspects_and_opinions
from Cluster_ASTE.BaseModel.Embedding import Word_embedding
from Cluster_ASTE.BaseModel.Cross_Transformer import Cross_Transformer
from Cluster_ASTE.ASTEModule.ATE_OTE import Get_aspect_opinion_index

nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')

#realize sentence paragraph division, segment
def Sentence_segment(sentence):
    #get the sentence constituent tree
    constituent_tree = nlp.parse(sentence)
    #normalization
    tree = Tree.fromstring(constituent_tree)

    #Get each segment sequence
    segment_positions = Get_paragraph_word_indices(sentence, tree)

    return segment_positions


#Realization of aspect-opinion pairing
def Aspect_opinion_pair(sentence):
    #Sentence paragraphing, segment
    segment_positions = Sentence_segment(sentence)
    # print(segment_positions)

    # Get aspect term index and opinion term index
    aspect_indices, opinion_indices = Get_aspect_opinion_index(sentence)
    # print('不使用多词感知:', aspect_indices, opinion_indices)

    # #Multi-word perception using dependencies
    # aspect_indices, opinion_indices = multword_Awareness(sentence)
    # print('使用多词感知:', aspect_indices, opinion_indices)

    # Merge sequences of positions together consecutively
    aspect_indices = group_consecutive_indices(aspect_indices)
    opinion_indices = group_consecutive_indices(opinion_indices)

    # Realize an aspect-opinion pair based on a sequence of paragraphs.
    pairs, segments = pair_aspects_and_opinions(aspect_indices, opinion_indices, segment_positions, )

    return pairs, segments


#Get the aspect_opinion and segment tensor and perform Cross Transformer
def Cross_computing(sentence, Cross_model):
    #Get pairs and paragraphs
    pairs, segments = Aspect_opinion_pair(sentence)

    #Get word feature vector
    text = nltk.word_tokenize(sentence)
    word_list, word_feature, pooled_embeddings = Word_embedding(text)

    # Stores the final_output computed for each pair
    final_outputs = []
    for pair, segment in zip(pairs, segments):
        # Extracting the position of aspect and opinion
        aspect_indices, opinion_indices = pair

        # Extracting aspect and opinion features and concat
        aspect_features = word_feature[aspect_indices]  # aspect_indices embedding
        opinion_features = word_feature[opinion_indices]  # opinion_indices embedding
        pair_feature = torch.cat([aspect_features, opinion_features], dim=0)

        # Get segment features
        segment_features = word_feature[segment]  # segment embedding

        # Cross_Transformer for final_output
        final_output = Cross_model(pair_feature, segment_features, segment_features)

        # Stores the computed final_output
        final_outputs.append(final_output)

    return pairs, final_outputs



###########################Test#################################
# sentence = "Speaking of the browser , it too has problems ."
# segment_positions = Sentence_segment(sentence)
# pairs, segments = Aspect_opinion_pair(sentence)
# print(pairs)
# print(segments)
#
# Cross_model = Cross_Transformer(768)
# pair, final_outputs = Cross_computing(sentence, Cross_model)
# print(pairs)  #[([16, 17], [15])]

# for idx, output in enumerate(final_outputs):
#     print(f"Final Output {idx} Shape: {output.shape}")   #torch.Size([3, 3])
#
#     # softmax output probability distribution to predict sentiment polarity
#     _, predicted_classes = torch.max(output, dim=-1)
#
#     # Predicted sentiment labeling
#     predicted_classes_list = predicted_classes.tolist()  # shape type list
#
#     # Statistical sentiment polarity frequency
#     counter = Counter(predicted_classes_list)
#
#     # Most frequent sentiment polarity labels
#     predict_label = counter.most_common(1)[0][0]





