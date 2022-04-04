import os

def get_bert_dir():
    return '../data/model/bert'


def get_model_dir():
    model_dir = '../data/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def get_chinese_wwm_ext_pytorch_path():
    bert_path = os.path.join(get_bert_dir(), 'chinese_wwm_ext_pytorch')
    return bert_path



def get_data_obj_dir():
    data_obj_dir = '../data/model/data_obj'
    if not os.path.exists(data_obj_dir):
        os.makedirs(data_obj_dir)
    return data_obj_dir


def get_word2id_path():
    word2id_path = os.path.join(get_data_obj_dir(), 'word2id.pkl')
    return word2id_path


def get_id2word_path():
    id2word_path = os.path.join(get_data_obj_dir(), 'id2word.pkl')
    return id2word_path


def get_tag2id_path():
    tag2id_path = os.path.join(get_data_obj_dir(), "tag2id.pkl")
    return tag2id_path


def get_pretrained_char_vec_path():
    pretained_char_vec_path = os.path.join(get_data_obj_dir(), "pretrained_char_vec.txt")
    return pretained_char_vec_path