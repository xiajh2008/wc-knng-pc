from sklearn import metrics
import numpy as np
import WC_KNNG_PC


if __name__ == '__main__':
    path = 'datasets/'
    # file_name = 'Aggregation.txt'
    # file_name = 'CMC.txt'
    # file_name = 'Compound.txt'
    # file_name = 'D31.txt'
    # file_name = 'flame.txt'
    # file_name = 'jain.txt'
    # file_name = 'pathbased.txt'
    # file_name = 'R15.txt'
    # file_name = 'spiral.txt'
    # file_name = 's2.txt'

    # file_name = 'ecoli.txt'
    # file_name = 'movement_libras.txt'
    # file_name = 'ionosphere.txt'
    file_name = 'iris.txt'
    # file_name = 'seeds.txt'
    # file_name = 'segmentation.txt'
    # file_name = 'wdbc.txt'
    # file_name = 'wine.txt'
    # file_name = 'spectrometer.txt'
    # file_name = 'glass.txt'
    # file_name = 'OlivettiFaces.txt'
    # file_name = 'usps.txt'
    # file_name = 'mnist.txt'

    data = np.loadtxt(path + file_name, delimiter=',')

    data_embedding = data[:, :-1]
    label = data[:, -1]

    """设置超参数"""
    t = 16  # 表示节点的t个最近邻
    k = 27  # 表示节点的k个最近邻

    """聚类"""
    center, cluster_nodes = WC_KNNG_PC.clustering(data_embedding, t, k)

    noise = [x for x in center if x < 0]
    ari = metrics.adjusted_rand_score(label, center)
    ami = metrics.adjusted_mutual_info_score(label, center)
    fmi = metrics.fowlkes_mallows_score(label, center)

    print('ARI: ', round(ari, 4), '\t', 'AMI: ', round(ami, 4), '\t', 'FMI: ', round(fmi, 4), '  noise ratio: ', round(len(noise) / len(label), 4))
