#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))

    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    # 记录语句与向量的对应关系
    centences_vec = {}
    for i, vec in enumerate(vectors):
        centences_vec[list(sentences)[i].replace(" ", "")] = vec

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # 计算每个簇的中心点与簇内所有句子的平均距离
    avg_to_center_dist = defaultdict(float)
    center_data = kmeans.cluster_centers_
    for i in range(len(center_data)):
        sentences_group = sentence_label_dict[i]
        for s in sentences_group:
            s = s.replace(" ", "")
            avg_to_center_dist[i] += np.sqrt(np.sum((center_data[i] - centences_vec[s])**2))
        avg_to_center_dist[i] = np.sqrt(avg_to_center_dist[i])

    print(avg_to_center_dist)
    sorted_dist = {k: v for k, v in sorted(avg_to_center_dist.items(), key=lambda item: item[1])}
    # 打印排序后的label及平均距离
    print(sorted_dist)

    # 打印前5个平均距离 最小的分类的前10个句子
    for label in list(sorted_dist.keys())[:5]:
        print("cluster %s :" % label)
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #打印10个
            print(list(sentences)[i].replace(" ", ""))
        print("---------")
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

