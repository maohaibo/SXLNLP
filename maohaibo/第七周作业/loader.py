# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import pandas as pd
import random
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        df = pd.read_csv(self.path)
        rows_as_lists = df.values.tolist()
        random.shuffle(rows_as_lists)
        for row in rows_as_lists:
            label = row[0]
            comment = row[1]
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(comment, max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(comment)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])

        # with open(self.path, encoding="utf8") as f:
        #     for line in f:
        #         line = json.loads(line)
        #         tag = line["tag"]
        #         label = self.label_to_index[tag]
        #         title = line["title"]
        #         if self.config["model_type"] == "bert":
        #             input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
        #         else:
        #             input_id = self.encode_sentence(title)
        #         input_id = torch.LongTensor(input_id)
        #         label_index = torch.LongTensor([label])
        #         self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    train_size = int(0.8 * len(dg))
    val_size = len(dg) - train_size
    train_dataset, val_dataset = random_split(dg, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=shuffle)

    # dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return train_loader, val_loader

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
