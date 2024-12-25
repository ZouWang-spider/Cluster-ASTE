

def Multword_term_aware(relation, graph, aspect_indices, opinion_indices):
    # Constructing a Relational Dictionary for Undirected Graphs
    relation_dict = {}
    for i, r in enumerate(relation):
        head, dependent = graph[0][i], graph[1][i]
        #  (head, relation, dependent) and (dependent, relation, head)
        if head not in relation_dict:
            relation_dict[head] = []
        if dependent not in relation_dict:
            relation_dict[dependent] = []
        relation_dict[head].append((dependent, r))
        relation_dict[dependent].append((head, r))

    # find aspect index
    modified_aspect_indices = set(aspect_indices)
    for idx in aspect_indices:
        if idx in relation_dict:
            for neighbor, rel in relation_dict[idx]:
                if rel in ['nn']:  # Check for a noun modification or direct object relation
                    modified_aspect_indices.add(neighbor)

    # find opinion index
    modified_opinion_indices = set(opinion_indices)
    for idx in opinion_indices:
        if idx in relation_dict:
            for neighbor, rel in relation_dict[idx]:
                if rel in ['advmod']:  # Check whether it is a modification relationship
                    modified_opinion_indices.add(neighbor)

    Aware_aspect_indices = sorted(modified_aspect_indices)
    Aware_opinion_indices = sorted(modified_opinion_indices)

    return Aware_aspect_indices, Aware_opinion_indices


#Extract words and consider multi-word phrases
def Extract_phrases_from_indices(sentence, aspect_indices, opinion_indices):

    words = sentence.split()

    # Function to extract phrases based on indices
    def get_phrases(indices):
        phrases = []
        current_phrase = []

        # Iterate over the given index
        for i, idx in enumerate(indices):
            # 如果当前索引是最后一个元素或与下一个元素不连续，则结束当前短语
            if i == len(indices) - 1 or indices[i] + 1 != indices[i + 1]:
                current_phrase.append(words[idx])
                phrases.append(" ".join(current_phrase))  # 将短语作为一个整体加入
                current_phrase = []  # 清空短语，开始下一个短语
            else:
                # 如果当前词与下一个词连续，继续将其合并
                current_phrase.append(words[idx])

        return phrases

    # 获取方面词和观点词的短语
    aspect_phrases = get_phrases(aspect_indices)
    opinion_phrases = get_phrases(opinion_indices)

    return aspect_phrases, opinion_phrases




############################Test#################################
# sentence = 'This beef burger is delicious.'
# aspect_indices = [2]
# opinion_indices = [4]
# relation = ['det', 'nn', 'nsubj', 'cop', 'root', 'punct']  # 依存关系
# graph = ([2, 2, 4, 4, 4, 4], [0, 1, 2, 3, 4, 5]) # 图的关系
#
# Aware_aspect_indices, Aware_opinion_indices = Multword_term_aware(relation, graph, aspect_indices, opinion_indices)
# print(Aware_aspect_indices)
# print(Aware_opinion_indices)