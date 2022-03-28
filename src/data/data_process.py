
def load_data(file_path):
    """
    # 加载数据
    word_lists: [[]]
    tag_lists: [[]]
    """
    word_lists = []
    tag_lists = []
    with open(file_path, 'r') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    return word_lists, tag_lists


def get_word2id(lists):
    """
    # 得到 word2id dict
    """
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


def extend_vocab(word2id, tag2id, for_crf=True):
    """
    # 补充word2id
    未登陆词: <unk>
    补码: <pad>
    句子开始标志: <start> 
    句子结束标志: <end>
    """
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def add_end_token(word_lists, tag_lists, test=False):
    '''
    # 加上结束符： <end>
    '''
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        # 给每个句子末尾加上 <end>
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists

