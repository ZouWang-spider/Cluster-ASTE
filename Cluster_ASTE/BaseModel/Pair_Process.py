

#将多词位置序列合并
def group_consecutive_indices(indices):
    if not indices:
        return []

    multi_term = []
    current_group = [indices[0]]  # 从第一个索引开始

    for i in range(1, len(indices)):
        # 判断当前索引是否连续
        if indices[i] == indices[i - 1] + 1:
            current_group.append(indices[i])
        else:
            # 如果不连续，则把当前组加入结果，并开始新的一组
            multi_term.append(current_group)
            current_group = [indices[i]]

    # 添加最后一组
    multi_term.append(current_group)

    return multi_term

#实现aspect-opinion pair和获取配对所在的段落
def pair_aspects_and_opinions(aspect_indices, opinion_indices, segment_positions):
    pairs = []
    segments = []

    for i, aspect_group in enumerate(aspect_indices):
        for j, opinion_group in enumerate(opinion_indices):
            # 查找是否存在重叠的段落
            # 假设每个段落都在segment_positions的一个位置里
            for segment in segment_positions:
                # 如果方面词和观点词的任意一个元素在该段落中都存在，则进行配对
                if any(index in segment for index in aspect_group) and any(index in segment for index in opinion_group):
                    pairs.append((aspect_group, opinion_group))  # 添加配对
                    segments.append(segment)  # 添加对应的段落
                    break  # 一旦找到匹配的段落，跳出循环

    # 如果没有找到任何配对,直接将aspect-opinion配对，并返回整个句子的index
    if not pairs:
        # 将所有的段落合并为一个段落
        merged_segment = [index for segment in segment_positions for index in segment]

        # 将当前所有的方面词和观点词配对
        for aspect_group in aspect_indices:
            for opinion_group in opinion_indices:
                pairs.append((aspect_group, opinion_group))  # 添加配对
        segments.append(merged_segment)  # 将整个句子视为一个段落


    return pairs, segments
