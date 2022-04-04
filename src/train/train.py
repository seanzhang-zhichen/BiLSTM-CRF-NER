
import os
import time
from data.data_process import load_data
from data.data_process import get_word2id
from data.data_process import extend_vocab
from data.data_process import add_end_token
from data.path import get_train_data_path
from data.path import get_eval_data_path
from data.path import get_test_data_path
from model.path import get_word2id_path
from model.path import get_tag2id_path
from model.path import get_id2word_path
from model.path import get_pretrained_char_vec_path
from tools.get_ner_level_acc import find_all_tag
from tools.help import save_as_pickle
from tools.help import load_pickle_obj
from tools.get_pretrained_vec import GetPretrainedVec
from .train_helper import NerModel

class Train:
    def __init__(self) -> None:
        self.train_data_path = get_train_data_path()
        self.eval_data_path = get_eval_data_path()
        self.test_data_path = get_test_data_path()
        self.word2id_path = get_word2id_path()
        self.tag2id_path = get_tag2id_path()
        self.id2word_path = get_id2word_path()
        self.vec_path = get_pretrained_char_vec_path()
        self.word2id = None
        self.tag2id = None
        self.id2word = None
        self.get_pretrained_vec = GetPretrainedVec()

    def load(self):
        self.get_pretrained_vec.load()

    def prepare_data(self):
        self.train_word_lists, self.train_tag_lists = load_data(self.train_data_path)
        self.eval_word_lists, self.eval_tag_list = load_data(self.eval_data_path)
        self.test_word_lists, self.test_tag_list = load_data(self.test_data_path)

        self.word2id = get_word2id(self.train_word_lists)
        self.tag2id = get_word2id(self.train_tag_lists)
        self.word2id, self.tag2id = extend_vocab(self.word2id, self.tag2id)

        self.id2word = {self.word2id[w]: w for w in self.word2id}

        save_as_pickle(self.word2id, self.word2id_path)
        save_as_pickle(self.tag2id, self.tag2id_path)
        save_as_pickle(self.id2word, self.id2word_path)
        
        if not os.path.exists(self.vec_path):
            print('用 BERT 生成预训练向量')
            self.get_pretrained_vec.get_bert_embed(self.train_data_path, self.vec_path, char=True)

        self.train_word_lists, self.train_tag_lists = add_end_token(self.train_word_lists, self.train_tag_lists)

        self.eval_word_lists, self.eval_tag_list = add_end_token(self.eval_word_lists, self.eval_tag_list)

        self.test_word_lists, self.test_tag_list = add_end_token(self.test_word_lists, self.test_tag_list, test=True)

        return (self.train_word_lists, self.train_tag_lists, 
                self.eval_word_lists, self.eval_tag_list, 
                self.test_word_lists, self.test_tag_list)


    def train(self, use_pretrained_w2v=False, model_type="bilstm-crf"):
        self.get_pretrained_vec.load()
        train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists = self.prepare_data()
        
        word2id = load_pickle_obj(self.word2id_path)
        tag2id = load_pickle_obj(self.tag2id_path)

        print(f"tag2id: {tag2id}")
        vocab_size = len(word2id)
        out_size = len(tag2id)
        ner_model = NerModel(vocab_size, out_size, use_pretrained_w2v=use_pretrained_w2v,  model_type=model_type)
        print(f"vocab_size: {vocab_size}, out_size: {out_size}")
        print("start to train the {} model ...".format(model_type))

        ner_model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists, word2id, tag2id)


    def predict(self, text, use_pretrained_w2v, model_type):
        word2id = load_pickle_obj(self.word2id_path)
        tag2id = load_pickle_obj(self.tag2id_path)
        vocab_size = len(word2id)
        out_size = len(tag2id)
        ner_model = NerModel(vocab_size, out_size, use_pretrained_w2v=use_pretrained_w2v,  model_type=model_type)
        result = ner_model.predict(text)
        return result

    def get_ner_list_dic(self, text, use_pretrained_w2v, model_type):

        text_list = list(text)
        tag_list = self.predict(text, use_pretrained_w2v, model_type)[0]
        tag_dic = find_all_tag(tag_list)

        print("tag_dic: ", tag_dic)

        result_dic = {}
        for name in tag_dic:
            for x in tag_dic[name]:
                if result_dic.get(name) is None:
                    result_dic[name] = []
                if x:
                    ner_name =  ''.join(text_list[x[0]:x[0]+x[1]])
                    result_dic[name].append(ner_name)
        for name in result_dic:
            result_dic[name] =  list(set(result_dic[name]))

        return result_dic