import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.get_pretrained_vec import GetPretrainedVec
from model.path import get_pretrained_char_vec_path
from model.path import get_word2id_path
from model.path import get_id2word_path


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, drop_out=0.5, use_pretrained_w2v=False):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        if use_pretrained_w2v:
            print("加载预训练字向量...")
            vec_path = get_pretrained_char_vec_path()
            word2id_path = get_word2id_path()
            id2word_path = get_id2word_path()
            embedding_pretrained = GetPretrainedVec.get_w2v_weight(emb_size, vec_path, word2id_path, id2word_path)
            self.embedding.weight.data.copy_(embedding_pretrained)
            self.embedding.weight.requires_grad = True
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, out_size)
        self.dropout = nn.Dropout(drop_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x, lengths):
        x = x.to(self.device)
        emb = self.embedding(x)
        emb = self.dropout(emb)
        emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        emb, _ = self.bilstm(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(emb, batch_first=True, padding_value=0., total_length=x.shape[1])
        scores = self.fc(output)
        return scores
    
    def predict(self, x, lengths, _):
        scores = self.forward(x, lengths)
        _, batch_tagids = torch.max(scores, dim=2)
        return batch_tagids


def cal_bilstm_loss(logits, targets, tag2id):
    PAD = tag2id.get('<pad>')
    assert PAD is not None
    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)
    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)
    return loss