import nltk

# from nltk import Tree
# from stanfordcorenlp import StanfordCoreNLP
# #Beef burger is fresh and delicious, and pizza is excellent.
# nlp = StanfordCoreNLP('D:/StanfordCoreNLP/stanford-corenlp-4.5.4')
#
# sentence ="The sushi seemed pretty fresh and was adequately proportioned ."
# constituent_tree = nlp.parse(sentence)
# # print(constituent_tree)
#
# # 将解析结果转换为 nltk.Tree
# tree = Tree.fromstring(constituent_tree)
# # 显示构成树
# tree.draw()
# # 关闭 StanfordCoreNLP
# nlp.close()


#出去重复的段落
def filter_redundant_segment(parsed_sentences):
    # 用于存储已添加的段落
    unique_segment = []

    # 遍历每个句子
    for sentence in parsed_sentences:
        # 遍历已存在的段落，检查是否已经包含当前段落
        is_redundant = False
        for unique_sentence in unique_segment:
            # 如果当前段落是已存在段落的子集，则认为是重复的
            if set(sentence).issubset(set(unique_sentence)):
                is_redundant = True
                break

        # 如果不是重复的，添加到 unique_sentences 中
        if not is_redundant:
            unique_segment.append(sentence)

    return unique_segment

#Stanford CoreNLP特殊字符处理
def restore_special_symbols(parsed_sentence):
    # 替换特殊符号
    restored_sentence = []
    for word in parsed_sentence:
        if word == '-LRB-':
            restored_sentence.append('(')
        elif word == '-RRB-':
            restored_sentence.append(')')
        elif word == '-LSB-':
            restored_sentence.append('[')
        elif word == '-RSB-':
            restored_sentence.append(']')
        elif word == '-LBR-':
            restored_sentence.append('{')
        elif word == '-RBR-':
            restored_sentence.append('}')
        elif word == '-HYPH-':
            restored_sentence.append('-')
        elif word == '-ELIPSIS-':
            restored_sentence.append('...')
        elif word == '-DASH-':
            restored_sentence.append('--')
        elif word == '-DOLLAR-':
            restored_sentence.append('$')
        else:
            restored_sentence.append(word)  # 不变的单词
    return restored_sentence

def process_sentences(sentences):
    restored_sentences = []
    for sentence in sentences:
        restored_sentences.append(restore_special_symbols(sentence))
    return restored_sentences



#获取段落的单词位置索引
def Get_paragraph_word_indices(sentence, tree):
    # 获取所有S节点的子树
    s_nodes = list(tree.subtrees(lambda t: t.label() == 'S'))
    # 分词
    words = nltk.word_tokenize(sentence)

    # 如果句子中只有一个S节点，直接返回所有单词的索引
    if len(s_nodes) == 1:
        return [list(range(len(words)))]  # 返回包含所有单词索引的列表

    # 否则获取每个S节点的单词位置索引
    s_words_indices = []
    for s_node in s_nodes:
        # 获取S节点包含的单词
        s_words = list(s_node.leaves())
        s_words_indices.append(s_words)
    s_words_indices = s_words_indices[1:]
    #除去重复的sgement
    s_words_indices = filter_redundant_segment(s_words_indices)
    #处理segment中的特殊字符
    s_words_indices = process_sentences(s_words_indices)

    # 存储所有段落的单词位置索引
    all_positions = []
    # 初始化last_element，表示当前段落的最后一个单词位置
    last_element = 0
    for segment in s_words_indices:
        # 获取当前段落的单词位置索引（从last_element位置开始搜索）
        segment_positions = [words.index(word, last_element) for word in segment]

        # 更新last_element，设置为当前段落的最后一个单词位置
        last_element = segment_positions[-1] + 1

        # 将当前段落的位置索引添加到all_positions中
        all_positions.append(segment_positions)

    #避免报错误，将无关单词存储在最后一个列表中作为段落
    total_words = len(words)
    all_indexes = set(range(total_words))
    # 获取现有段落的所有已使用的索引
    used_indexes = set()
    for segment in all_positions:
        used_indexes.update(segment)

    # 计算句子中未出现的索引（即句子总索引减去已使用的索引）
    missing_indexes = list(all_indexes - used_indexes)
    # 将缺失的索引添加到 index 的最后一个段落
    all_positions.append(missing_indexes)

    return all_positions

