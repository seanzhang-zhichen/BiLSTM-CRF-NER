# coding = utf-8
import jieba
import torch
import logging
import numpy as np
from tqdm import tqdm
from .help import load_pickle_obj
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizerFast
from model.path import get_chinese_wwm_ext_pytorch_path


jieba.setLogLevel(logging.INFO)


class GetPretrainedVec:
    def __init__(self):
        self.bert_path = get_chinese_wwm_ext_pytorch_path()

    def load(self):
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.token = BertTokenizerFast.from_pretrained(self.bert_path)

    # Bert 字向量生成
    def get_data(self, path, char=False):
        words = []
        with open(path, "r", encoding="utf-8") as f:
            sentences = f.readlines()
            if char:
                for sent in sentences:
                    words.extend([word.strip() for word in sent.strip().replace(" ", "") if word not in words])
            else:
                for sentence in sentences:
                    cut_word = jieba.lcut(sentence.strip().replace(" ", ""))
                    words.extend([w for w in cut_word if w not in words])
        return words


    def get_bert_embed(self, src_path, vec_save_path, char=False):
        words = self.get_data(src_path, char)
        words.append("<unk>")
        words.append("<pad>")
        words.append("<start>")
        words.append("<end>")
        # 字向量
        if char:
            file_char = open(vec_save_path, "a+", encoding="utf-8")
            file_char.write(str(len(words)) + " " + "768" + "\n")
            for word in tqdm(words, desc="编码字向量:"):
                inputs = self.token.encode_plus(word, padding="max_length", truncation=True, max_length=10,
                                        add_special_tokens=True,
                                        return_tensors="pt")
                out = self.bert(**inputs)
                out = out[0].detach().numpy().tolist()
                out_str = " ".join("%s" % embed for embed in out[0][1])
                embed_out = word + " " + out_str + "\n"
                file_char.write(embed_out)
            file_char.close()
        else:
            file_word = open(vec_save_path, "a+", encoding="utf-8")
            file_word.write(str(len(words)) + " " + "768" + "\n")
            # 词向量 (采用字向量累加求均值)
            for word in tqdm(words, desc="编码词向量:"):
                words_embed = np.zeros(768)  # bert tensor shape is 768
                inputs = self.token.encode_plus(word, padding="max_length", truncation=True, max_length=50, add_special_tokens=True,
                                        return_tensors="pt")
                out = self.bert(**inputs)
                word_len = len(word)
                out_ = out[0].detach().numpy()
                for i in range(1, word_len + 1):
                    out_str = out_[0][i]
                    words_embed += out_str
                words_embed = words_embed / word_len
                words_embedding = words_embed.tolist()
                result = word + " " + " ".join("%s" % embed for embed in words_embedding) + "\n"
                file_word.write(result)

            file_word.close()
    
    @staticmethod
    def get_w2v_weight(embedding_size, vec_path, word2id_path, id2word_path):
        w2v_model = KeyedVectors.load_word2vec_format(vec_path, binary=False)
        
        word2id = load_pickle_obj(word2id_path)
        id2word = load_pickle_obj(id2word_path)
        vocab_size = len(word2id)
        embedding_size = embedding_size
        weight = torch.zeros(vocab_size, embedding_size)
        for i in range(len(w2v_model.index2word)):
            try:
                index = word2id[w2v_model.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(w2v_model.get_vector(id2word[word2id[w2v_model.index2word[i]]]))

        return weight

