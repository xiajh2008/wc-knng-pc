import numpy as np
from numpy import loadtxt, ndarray, unique
from collections import Counter
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from sklearn import preprocessing
import copy
import math
import warnings
import matplotlib.cbook
import itertools
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)  # 精确表示小数
np.seterr(invalid='ignore')



if __name__ == '__main__':
    path = 'Data sets/'
    # file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'


    """SETTING PARAMETERS"""
    t = 9  # 表示节点的t个最近邻
    k = 19  # 表示节点的k个最近邻

    """READ DATA"""
    data = np.loadtxt(path + file_name, delimiter=',')
    data_embedding = data[:, :-1]
    label = data[:, -1]

    import wc_knng_pc
    assert t <= k < data_embedding.shape[0]
    predict = wc_knng_pc.clustering(t, k, data_embedding)

    noise = [x for x in predict if x < 0]
    truth = label
    ari = metrics.adjusted_rand_score(truth, predict)
    ami = metrics.adjusted_mutual_info_score(truth, predict)
    fmi = metrics.fowlkes_mallows_score(truth, predict)
    nmi = metrics.normalized_mutual_info_score(truth, predict)

    print(f'ari: {round(ari, 3)}, \tami: {round(ami, 3)}, \tnmi: {round(nmi, 3)}, \tfmi: {round(fmi, 3)}')