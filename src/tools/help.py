import os
import torch
import pickle


def save_as_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def sort_by_lengths(word_lists,tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)), key=lambda x: len(pairs[x][0]), reverse=True)

    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists, indices


def batch_sents_to_tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)

    lengths = [len(l) for l in batch]
    return batch_tensor, lengths


def flatten_lists(lists):
    """将list of list 压平成list"""
    flatten_list = []
    for list_ in lists:
        if type(list_) == list:
            flatten_list.extend(list_)
        else:
            flatten_list.append(list_)
    return flatten_list