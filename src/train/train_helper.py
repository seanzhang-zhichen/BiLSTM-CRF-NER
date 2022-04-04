import json
import os
import torch

import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm, trange
from model.bilstm import BiLSTM, cal_bilstm_loss
from model.bilstm_crf import BiLSTM_CRF, cal_bilstm_crf_loss
from tools.get_ner_level_acc import precision
from tools.help import sort_by_lengths, batch_sents_to_tensorized
from model.path import get_model_dir

class NerModel(object):
    def __init__(self, vocab_size, out_size, use_pretrained_w2v=False, model_type="bilstm-crf"):
        self.model_dir = get_model_dir()
        self.model_type_dir = os.path.join(self.model_dir, model_type)
        if not os.path.exists(self.model_type_dir):
            os.makedirs(self.model_type_dir)
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.out_size = out_size
        self.batch_size = 64
        self.lr = 0.01
        self.epoches = 10
        self.print_step = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = 768
        self.hidden_size = 256
        self.dropout = 0.5
        self.use_pretrained_w2v = use_pretrained_w2v
        if self.model_type == "bilstm-crf":
            self.model = BiLSTM_CRF(self.vocab_size, self.emb_size, self.hidden_size, self.out_size, self.dropout, self.use_pretrained_w2v)
            self.loss_cal_fun = cal_bilstm_crf_loss
        elif self.model_type == "bilstm":
            self.model = BiLSTM(self.vocab_size, self.emb_size, self.hidden_size, self.out_size, self.dropout, self.use_pretrained_w2v)
            self.loss_cal_fun = cal_bilstm_loss
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=0.005)
        self.step = 0
        self.best_val_loss = 1e18
    
    def train(self, train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists, word2id, tag2id):
        train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        total_step = (len(train_word_lists)//self.batch_size + 1)

        epoch_iterator = trange(1, self.epoches + 1, desc="Epoch")
        for epoch in epoch_iterator:
            self.step = 0
            loss_sum = 0.
            for idx in trange(0, len(train_word_lists), self.batch_size, desc="Iteration:"):
                batch_sents = train_word_lists[idx : idx + self.batch_size]
                batch_tags = train_tag_lists[idx : idx + self.batch_size]
                loss_sum += self.train_step(batch_sents, batch_tags, word2id, tag2id)
                if self.step == total_step:
                    print("\nEpoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        loss_sum / self.print_step
                    ))
                    loss_sum = 0.
            self.validate(epoch, dev_word_lists, dev_tag_lists, word2id, tag2id)
            

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step+=1
        batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(batch_sents, word2id)
        labels_tensor, _ = batch_sents_to_tensorized(batch_tags, tag2id)

        batch_sents_tensor, labels_tensor = batch_sents_tensor.to(self.device), labels_tensor.to(self.device)
        scores = self.model(batch_sents_tensor, sents_lengths)

        self.model.zero_grad()
        loss = self.loss_cal_fun(scores, labels_tensor, tag2id)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    
    def validate(self, epoch, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_step = 0
            for idx in range(0, len(dev_word_lists), self.batch_size):

                val_step+=1
                batch_sents = dev_word_lists[idx : idx + self.batch_size]
                batch_tags = dev_tag_lists[idx : idx + self.batch_size]
                batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(batch_sents, word2id)
                labels_tensor, _ = batch_sents_to_tensorized(batch_tags, tag2id)
                batch_sents_tensor, labels_tensor = batch_sents_tensor.to(self.device), labels_tensor.to(self.device)
                scores = self.model(batch_sents_tensor, sents_lengths)

                loss = self.loss_cal_fun(scores, labels_tensor, tag2id).item()

                val_loss += loss

            print(f"------epoch: {epoch}, val loss: {val_loss}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                if self.use_pretrained_w2v:
                    model_name = f"{self.model_type}-pretrained.pt"
                else:
                    model_name = f"{self.model_type}.pt"
                model_save_path = os.path.join(self.model_type_dir, model_name)
                print(f"保存模型，path: {model_save_path}")
                torch.save(self.best_model.state_dict(), model_save_path)
                print(f"curren best val loss: {self.best_val_loss}")

    def test(self, test_word_lists, test_tag_lists, word2id, tag2id):
        test_word_lists,test_tag_lists,indices = sort_by_lengths(test_word_lists, test_tag_lists)
        batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(test_word_lists, word2id)
        batch_sents_tensor = batch_sents_tensor.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.predict(batch_sents_tensor, sents_lengths, tag2id)
        pre_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.model_type == "bilstm-crf":
                for j in range(sents_lengths[i] - 1):
                    tag_list.append(id2tag[ids[j]].item())
            else:
                for j in range(sents_lengths[i]):
                    tag_list.append(id2tag[ids[j]].item())  
            pre_tag_lists.append(tag_list)           
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        total_precision, result_dic = precision(pred_tag_lists, tag_lists)
        print(f"实体级准确率为: {total_precision}")
        print(f"各实体对应的准确率为: {json.dumps(result_dic, ensure_ascii=False, indent=4)}")
