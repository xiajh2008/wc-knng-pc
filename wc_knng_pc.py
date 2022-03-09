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


def read_dataset(data_embedding):  # 用于s1,s2,s3,s4文件数据的读取

    cluster_label = list()

    pos = dict()

    node_dict = dict()
    data_size = data_embedding.shape[0]
    for i in range(data_size):
        cluster_label.append(i)
        node_dict[i] = len(node_dict)
        pos[i] = data_embedding[i].tolist()  # 用于可视化的字典

    # print("The size and shape of dataset are: ", data_size, data_embedding.shape)

    return cluster_label, pos, node_dict, data_size


def os_distance(data_size, data_embedding, cluster_label, knn, knn_large):  # 计算节点之间的欧氏距离
    knn_values, knn_nodes, knn_values_large, knn_nodes_large = list(), list(), list(), list()
    for i in range(data_size):
        temp = np.linalg.norm(data_embedding - data_embedding[i], axis=1, keepdims=True).reshape(1, -1)
        simi_list = temp[0]
        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
        klarge_points = [cluster_label[simi_sorted[i]] for i in range(knn_large + 1)]
        klarge_values = [simi_list[simi_sorted[i]] for i in range(knn_large + 1)]

        indi = klarge_points.index(cluster_label[i])
        klarge_points.pop(indi)
        klarge_values.pop(indi)

        knn_nodes_large.append(klarge_points)
        knn_values_large.append(klarge_values)

        knn_nodes.append(klarge_points[:knn])
        knn_values.append(klarge_values[:knn])

    return knn_nodes, knn_values, knn_nodes_large, knn_values_large


def point_td_distance(data_size, cluster_label, pos, knn_nodes, knn_values, node_dict):
    knn_values = np.array(knn_values)
    weight_0 = knn_values / np.sum(knn_values, axis=1, keepdims=True)  # 大小为： 点数 × KNN

    average0 = list()  # 保存Ai与Ai的KNN之间的平均距离
    for term0 in cluster_label:  # 读取所有节点中的一个节点
        average1 = list()
        for term1 in knn_nodes[node_dict[term0]]:  # 一个节点的knn中的每个节点
            shares = list(set(knn_nodes[node_dict[term0]]).intersection(knn_nodes[node_dict[term1]]))  # 当前节点与它的knn中每个节点的交集

            if len(shares) > 0:
                sum0 = 0.0
                for term2 in shares:
                    sum0 = sum0 + np.linalg.norm(np.array(pos[term0]) - np.array(pos[term2])) \
                           + np.linalg.norm(np.array(pos[term1]) - np.array(pos[term2]))  # dist(Ai-Xi)+dist(Aj-Xi)
                average1.append(sum0 / len(shares))  # 节点Ai与KNN中的某一个节点之间的共享近邻距离和/共享近邻点数，即共享近邻平均距离
            else:
                average1.append(2 * np.linalg.norm(np.array(pos[term0]) - np.array(pos[term1])))  # 用 dist(Ai-Aj)
                # 来构成一个更长的距离，说明该点距离Ai更远
        average0.append(average1)

    weight_0 = np.array(weight_0)
    average0 = np.array(average0)

    td_distance = list()
    for i in range(data_size):
        td_distance.append(np.dot(weight_0[i], average0[i]))

    return td_distance, pos


def data_preprocessing(data_size, node_dict, knn, knn_nodes, knn_large, knn_nodes_large, cluster_label, td_distance, kesai, kesai_large, knn_values, knn_values_large, sigmak):
    """＊＊＊＊＊＊＊＊step1: 计算每个点与其knn的共享近邻数量， 归一化所有点的knn的共享连通距离期望＊＊＊＊＊＊＊＊"""
    global temp0
    """小2度距离：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：："""
    knn_td_distance = [list() for i in range(data_size)]
    knn_fc_td_distance = [list() for i in range(data_size)]
    knn_mean_td_distance = [list() for i in range(data_size)]

    for i in range(data_size):
        knn_td_distance[i].append(td_distance[i])
        for j in range(knn_large):
            knn_td_distance[i].append(td_distance[node_dict[knn_nodes_large[i][j]]])
        knn_mean_td_distance[i].append(float(np.mean(np.array(knn_td_distance[i][:knn + 1]))))  # 只用前小knn个节点的共享距离期望来获得当前点的均值和方差
        knn_fc_td_distance[i].append(float(np.std(np.array(knn_td_distance[i][:knn + 1]), ddof=1)))  # 只用前小knn个节点的共享距离期望来获得当前点的均值和方差
        knn_td_distance[i] = knn_td_distance[i][1:]  # 去除第0个自己的共享距离

    distance_mean = [list() for i in range(data_size)]
    distance_std = [list() for i in range(data_size)]
    for i in range(data_size):
        distance_mean[i].extend([knn_mean_td_distance[node_dict[item]][0] for item in knn_nodes_large[i]])
        distance_std[i].extend([knn_fc_td_distance[node_dict[item]][0] for item in knn_nodes_large[i]])

    """×××××××××××××××××××××××××××××××××××××××××计算所有点的2sigma准则×××××××××××××××××××××××××××××××××××××××××××"""
    sigmak2 = [list() for i in range(data_size)]
    sigmak2_inve = [list() for i in range(data_size)]
    for i in range(data_size):
        for j in range(knn_large):
            nn_label = knn_nodes_large[i][j]
            nn_index = node_dict[nn_label]
            if abs(td_distance[nn_index] - knn_mean_td_distance[i][0]) <= sigmak * knn_fc_td_distance[i][0]:  # 近邻点与当前点的偏差sigma3
                sigmak2_inve[i].append(1)
            else:
                sigmak2_inve[i].append(0)

            if abs(td_distance[i] - distance_mean[i][j]) <= sigmak * distance_std[i][j]:  # 当前点与近邻点的偏差sigma3
                sigmak2[i].append(1)
            else:
                sigmak2[i].append(0)

    """×××××××××××××××××××××××××××××××××××××××××计算点P与KNN每个点的SNN数量×××××××××××××××××××××××××××××××××××××××××××"""
    shares_num = [list() for i in range(data_size)]  # 保存每个节点与knn之间共享近邻数量
    shares_num_max = [list() for i in range(data_size)]
    for i in range(data_size):
        temp_share = list()
        for item in knn_nodes_large[i]:
            length = len(set(knn_nodes[i]).intersection(knn_nodes[node_dict[item]]))  # 用大knn来计算共享近邻的交集
            length_max = len(set(knn_nodes_large[i]).intersection(knn_nodes_large[node_dict[item]]))  # 用大knn_large来计算共享近邻的交集

            shares_num[i].append(length)
            shares_num_max[i].append(length_max)

            if length_max >= kesai_large:
                temp_share.append(1)
            else:
                temp_share.append(0)

    '''×××××××××××××××××××××××××××××××××××××××××计算density-td_distance值，衡量点的knn中满足三条件的点2度距离，单位距离包含的点数量×××××××××××××××××××××××××××××××××××××××××××'''
    density_td_distance = list()
    density_number = list()
    sigma_hold = list()
    for i in range(data_size):
        number, number1 = 0, 0
        os_sum = list()
        for j in range(knn):
            os_sum.append(knn_values[i][j])
            if sigmak2[i][j] == 1 and sigmak2_inve[i][j] == 1 and shares_num[i][j] >= kesai:
                number += 1
                number1 += 1

        sigma_hold.append(number1 / knn)
        density_number.append(number)
        density_td_distance.append(number / np.mean(np.array(os_sum)))  # 2度距离的密度值

    '''×××××××××××××××××××××××××××计算点的调优海拔值×××××××××××××××××××××××××××××××××××××××××××××××'''
    density_td_distance = preprocessing.minmax_scale(density_td_distance)
    altitude = list()
    for i in range(len(cluster_label)):
        altitude.append(td_distance[i] * math.exp(-(sigma_hold[i] * density_td_distance[i])))

    '''#####################################二次计算sigmak2, simgak2_inve########################################################'''
    del knn_td_distance, knn_fc_td_distance, knn_mean_td_distance, distance_mean, distance_std, sigmak2, sigmak2_inve, density_td_distance, density_number, sigma_hold
    '''清除不需要的变量'''
    knn_altitude = [list() for i in range(data_size)]
    knn_mean_altitude = [list() for i in range(data_size)]
    knn_fc_altitude = [list() for i in range(data_size)]
    for i in range(data_size):
        knn_altitude[i].append(altitude[i])
        for j in range(knn_large):
            knn_altitude[i].append(altitude[node_dict[knn_nodes_large[i][j]]])
        knn_mean_altitude[i].append(float(np.mean(np.array(knn_altitude[i][:knn + 1]))))  # 只用前小knn个节点的共享距离期望来获得当前点的均值和方差
        knn_fc_altitude[i].append(float(np.std(np.array(knn_altitude[i][:knn + 1]), ddof=1)))  # 只用前小knn个节点的共享距离期望来获得当前点的均值和方差

        knn_altitude[i] = knn_altitude[i][1:]  # 去除第0个自己的共享距离

    knn_mean = [list() for i in range(data_size)]
    knn_std = [list() for i in range(data_size)]
    for i in range(data_size):
        knn_mean[i].extend([knn_mean_altitude[node_dict[item]][0] for item in knn_nodes_large[i]])
        knn_std[i].extend([knn_fc_altitude[node_dict[item]][0] for item in knn_nodes_large[i]])

    sigmak2 = [list() for i in range(data_size)]
    sigmak2_inve = [list() for i in range(data_size)]
    for i in range(data_size):
        for j in range(knn_large):
            nn_label = knn_nodes_large[i][j]
            nn_index = node_dict[nn_label]
            if abs(altitude[nn_index] - knn_mean_altitude[i][0]) <= sigmak * knn_fc_altitude[i][0]:  # 近邻点与当前点的偏差sigma3准则
                sigmak2_inve[i].append(1)
            else:
                sigmak2_inve[i].append(0)

            if abs(altitude[i] - knn_mean[i][j]) <= sigmak * knn_std[i][j]:  # 当前点与近邻点的偏差sigma3准则
                sigmak2[i].append(1)
            else:
                sigmak2[i].append(0)

    density_altitude = list()
    density_number = list()
    sigma_hold = list()
    indegree = list()
    stable = list()
    non_stable_one = list()
    for i in range(data_size):
        number, number1 = 0, 0
        os_sum = list()
        for j in range(knn):
            os_sum.append(knn_values[i][j])
            if sigmak2[i][j] == 1 and sigmak2_inve[i][j] == 1 and shares_num[i][j] >= kesai:
                number += 1
                number1 += 1

        density_number.append(number)
        sigma_hold.append(number1 / knn)
        density_altitude.append(number / np.mean(np.array(os_sum)))  # 2度距离的密度值

        degree_num = 0
        for x in range(knn):
            if sigmak2[i][x] == 1 and sigmak2_inve[i][x] == 1 and shares_num[i][x] >= kesai:
                degree_num += 1
        indegree.append(degree_num)  # 符合准则的近邻点数量
        stable.append(degree_num / knn)
        if 0 < degree_num / knn < 0.2:
            non_stable_one.append(cluster_label[i])  # 保存极不稳定的节点

    anomalies = list()
    non_stable_zero = list()
    for i in range(data_size):  # 满足海拔的sigma2准则的点密度
        if indegree[i] == 0:
            non_stable_zero.append(cluster_label[i])
        else:
            temp0 = list()
            temp0.append(cluster_label[i])
            temp0.extend(knn_nodes_large[i])
            temp0 = [density_altitude[node_dict[x]] for x in temp0]
            if density_altitude[i] == min(temp0):
                anomalies.append(cluster_label[i])

    return altitude, knn_altitude, knn_mean, knn_std, shares_num, shares_num_max, sigmak2, sigmak2_inve, sigma_hold, indegree, stable, non_stable_one, anomalies, non_stable_zero


def cluster_os_distance(data_embedding, node_dict, cluster_nodes, cluster_nodes_d, cluster_size, knn, cluster_inve_dict, vote_in, indegree):
    """计算簇内每个点的[k/2]nn, knn_in_cluster保存每个簇内的每个点的knn。
    如果点x为核心点，则从knn个核心点; 如果点x为非核心点，则取knn个距离中心点距离小于等于x到中心点的距离"""
    kk = knn
    if kk < 2:
        kk = 2
    knn_in_cluster = [list() for i in range(cluster_size)]
    for i in range(cluster_size):
        """簇核心区域点集"""
        core = cluster_inve_dict[i]  # 簇的中心点
        core_e = data_embedding[node_dict[core]]
        core_points = [x for x in cluster_nodes[i] if vote_in[node_dict[x]][0] == cluster_inve_dict[i] and x != core]  # 中心区域点

        """汇集一个簇的点的向量集"""
        data_vector_cluster, core_data_vector_cluster = list(), list()
        for x in cluster_nodes_d[i]:  # 使用cluster_nodes_d获取簇数据嵌入矩阵
            data_vector_cluster.append(data_embedding[node_dict[x]].tolist())
        data_vector_cluster = np.array(data_vector_cluster)

        # subc_length = len(cluster_nodes[i])
        subc_length_d = len(cluster_nodes_d[i])

        if subc_length_d <= kk + 1:
            """如果簇内点数量小于knn，则取簇的所有点作为每个点的最近邻; 否则只取前knn个簇内点"""
            for j in range(subc_length_d):
                if cluster_nodes_d[i][j] in cluster_nodes[i]:  # 子簇内的真实数据点
                    temp = copy.copy(cluster_nodes_d[i])
                    temp.remove(cluster_nodes_d[i][j])
                    knn_in_cluster[i].append(temp)  # 注意判断后面的knn_in_cluster的各元素的长度
        else:
            """如果点x为核心点，则选择簇内knn个点; 如果点x为非核心点，则取knn个距离中心点距离小于等于x到中心点的距离"""
            for j in range(subc_length_d):
                if cluster_nodes_d[i][j] in cluster_nodes[i]:  # 子簇内的真实数据点
                    """首先满足簇内的真实点，而不是簇的扩展集的点"""
                    indice_node = node_dict[cluster_nodes_d[i][j]]  # 扩展簇的点
                    if cluster_nodes_d[i][j] in core_points or cluster_nodes_d[i][j] == core or indegree[indice_node] / knn >= 0.6:
                        temp = np.linalg.norm(data_vector_cluster - data_vector_cluster[j], axis=1, keepdims=True).reshape(1, -1)
                        simi_list = temp[0]
                        simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
                        temp = [cluster_nodes_d[i][simi_sorted[k]] for k in range(1, kk + 1)]  # 得到普通knn
                        knn_in_cluster[i].append(temp)  # 0表示点自己，1表示第1个近邻
                    else:
                        knode, kvv = list(), list()
                        v1 = core_e - data_vector_cluster[j]  # 取向量V1
                        for k in range(subc_length_d):
                            if k != j:
                                v2 = data_vector_cluster[k] - data_vector_cluster[j]
                                if np.dot(v1, v2) >= 0:
                                    knode.append(cluster_nodes_d[i][k])
                                    kvv.append(data_vector_cluster[k].tolist())
                        if len(kvv) >= kk:
                            """如果往中心点方向的点数量满足knn数量，则从中选择"""
                            kvv = np.array(kvv)
                            temp = np.linalg.norm(kvv - data_vector_cluster[j], axis=1, keepdims=True).reshape(1, -1)
                            simi_list = temp[0]
                            simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
                            knn_in_cluster[i].append([knode[simi_sorted[k]] for k in range(kk)])  # 0表示点自己，1表示第1个近邻
                        else:
                            """从簇内选择knn"""
                            temp = np.linalg.norm(data_vector_cluster - data_vector_cluster[j], axis=1, keepdims=True).reshape(1, -1)
                            simi_list = temp[0]
                            simi_sorted = np.argsort(simi_list)  # 按欧氏距离升序排序
                            knn_in_cluster[i].append([cluster_nodes_d[i][simi_sorted[k]] for k in range(1, kk + 1)])  # 0表示点自己，1表示第1个近邻
    return knn_in_cluster


def pc_span(cluster_size, cluster_inve_dict, node_dict, cluster_nodes, vote_in, data_embedding, knn_in_cluster, knn_nodes, data_size, cluster_anomalies):
    """计算所有子簇的点平均跨度和中心区的点平均跨度, 返回参数： avg_core, avg_subc, cp_span, ck_span--------------------------"""
    # avg_core = [0 for i in range(cluster_size)]
    # avg_subc = [0 for i in range(cluster_size)]
    cp_span = [list() for i in range(cluster_size)]
    span_nodes = [-1 for i in range(data_size)]

    norm_subc = [list() for i in range(cluster_size)]

    for i in range(cluster_size):
        """簇的中心点"""
        core = cluster_inve_dict[i]
        core_e = data_embedding[node_dict[core]]
        """两种点：中心点和其他点"""
        core_points = [x for x in cluster_nodes[i] if vote_in[node_dict[x]][0] == cluster_inve_dict[i] and x != core]  # 中心区域点
        subc_length = len(cluster_nodes[i])

        """长度>2的子簇的度的计算"""
        for j in range(subc_length):
            """长度<2的knn的跨度"""
            if len(knn_in_cluster[i][j]) < 2:
                """norm_subc是按子簇保存，span_nodes按点字典顺序保存"""
                cp_span[i].append(0)
                # norm_subc[i].append(0)
                # span_nodes[node_dict[cluster_nodes[i][j]]] = 0
            else:
                v0 = data_embedding[node_dict[cluster_nodes[i][j]]]
                comb = itertools.combinations(knn_in_cluster[i][j], 2)
                ts = list()
                first_fz = 0
                first_fm = 0
                seccond_fz = 0
                seccond_fm = 0
                for s in comb:
                    indice1 = node_dict[s[0]]
                    indice2 = node_dict[s[1]]

                    v1 = data_embedding[indice1] - v0
                    v2 = data_embedding[indice2] - v0
                    first_fz += np.square(np.dot(v1, v2) / (np.square(np.linalg.norm(v1)) * np.square(np.linalg.norm(v2)))) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    first_fm += 1 / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    seccond_fz += (np.dot(v1, v2) / (np.square(np.linalg.norm(v1)) * np.square(np.linalg.norm(v2)))) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    seccond_fm += 1 / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    """簇的每个点的跨度"""
                point_span = first_fz / first_fm - np.square(seccond_fz / seccond_fm)
                cp_span[i].append(point_span)

        """完成一个子簇内所有点的跨度的计算，再进行最大值归一化。由于每个子簇的形成会造成一个大簇的过度分裂，出现过多的0跨度点，所以，也可以使用0跨度点的减少来作为内部评价指标"""
        span_max = np.max(np.array(cp_span[i]))
        if span_max == 0:
            for j in range(subc_length):
                norm_subc[i].append(0)
                span_nodes[node_dict[cluster_nodes[i][j]]] = 0
        else:
            for j in range(subc_length):
                """最大值归一化"""
                ap_span = cp_span[i][j] / span_max
                norm_subc[i].append(ap_span)
                span_nodes[node_dict[cluster_nodes[i][j]]] = ap_span

    """不可信的扩展点：如果点x为子簇的奇异点，且跨度<0.2，则被列为不可信的扩展点;如果点y为x的knn，且跨度<0.2也被列为不可信的扩展点"""
    non_extendable = list()
    s = list()
    for i in range(cluster_size):
        if len(cluster_anomalies[i]) > 0:
            s.extend(cluster_anomalies[i])
            for x in cluster_anomalies[i]:
                if span_nodes[node_dict[x]] < 0.2:
                    non_extendable.append(x)
                    for y in knn_nodes[node_dict[x]]:
                        if span_nodes[node_dict[y]] < 0.2 and not (y in non_extendable):
                            non_extendable.append(y)
    return cp_span, norm_subc, span_nodes, non_extendable


def degree_of_belonging(label_center, label_host, label_nn, node_dict, cluster_dict, cluster_nodes, cluster_label, sigmak2, sigmak2_inve, shares_num_max, knn, knn_nodes_large, knn_values_large, kesai_large):
    dbv = -1
    same, diff = 0, 0
    tid = node_dict[label_nn]  # 保存label的字典索引的临时变量
    """计算当前被连接点的归属系数DBV"""
    if label_host in knn_nodes_large[tid]:
        hid = knn_nodes_large[tid].index(label_host)  # 主点在近邻点的KNN的索引位置
        for i in range(hid + 1):
            if knn_nodes_large[tid][i] in cluster_nodes[cluster_dict[label_center]]:  # 如果label_nn的knn的某个点是否与label_host属于同一个局部集水盆
                if sigmak2[tid][i] == 1 and sigmak2_inve[tid][i] == 1 and shares_num_max[tid][i] >= kesai_large:
                    same += 1  # positive vote  从点的kNN的前hid+1中属于主点的同一个类中有多少点同意从点加入，正投票
                else:
                    diff += 1  # negative vote  从点的kNN的前hid+1中属于主点的同一个类中有多少点不同意从点加入，负投票
        dbv = same - diff
    return dbv


def first_clustering(index_host, index_nn, label_host, label_nn, center, altitude, cluster_label, label_center, cluster_dict, cluster_nodes, cluster_nodes_d, edges, vote_in, vote_out, anomalies, cluster_belinked, cluster_anomalies, node_dict, sigmak2, sigmak2_inve, shares_num_max, knn, knn_nodes,
                     knn_nodes_large, knn_values_large, kesai_large, stable):
    label_nn_root = cluster_label[center[index_nn]]
    label_host_root = cluster_label[center[index_host]]

    if altitude[index_host] <= altitude[index_nn]:
        outward_jd = label_host
        outward_md = altitude[index_host]
    else:
        outward_jd = label_nn
        outward_md = altitude[index_nn]

    ccid = cluster_dict[label_center]
    if len(vote_in[index_nn]) < 1:  # ----------- 孤立点
        if label_nn in knn_nodes:
            center[index_nn] = center[index_host]  # 中心节点为父节点的父节点标签
            cluster_nodes[ccid].append(label_nn)  # 把可连接的近邻点添加到局部簇或全局簇, 每个值小的节点都有可能成为簇的中心，拥有一个列表
            cluster_nodes_d[ccid].append(label_nn)  # 这里内容操作与cluster_nodes相同

            edges.append((label_host, label_nn))  # 添加边
            vote_in[index_nn].append(label_host)  # 中心点为其入度点，这里是有向边
            vote_out[index_host].append(label_nn)
            if label_nn in anomalies:  # 判断是否为奇异点,不能被重复添加，
                if not (label_nn in cluster_anomalies[ccid]):
                    cluster_anomalies[ccid].append(label_nn)  # 保存簇的奇异点

        else:
            if degree_of_belonging(label_center, label_host, label_nn, node_dict, cluster_dict, cluster_nodes, cluster_label, sigmak2, sigmak2_inve, shares_num_max, knn, knn_nodes_large, knn_values_large, kesai_large) > 0:
                center[index_nn] = center[index_host]  # 中心节点为父节点的父节点标签
                cluster_nodes[ccid].append(label_nn)  # 把可连接的近邻点添加到局部簇或全局簇, 每个值小的节点都有可能成为簇的中心，拥有一个列表
                cluster_nodes_d[ccid].append(label_nn)  # 这里内容操作与cluster_nodes相同

                edges.append((label_host, label_nn))  # 添加边
                vote_in[index_nn].append(label_host)  # 中心点为其入度点，这里是有向边
                vote_out[index_host].append(label_nn)

                if label_nn in anomalies:  # 判断是否为奇异点,不能被重复添加，
                    if not (label_nn in cluster_anomalies[ccid]):
                        cluster_anomalies[ccid].append(label_nn)  # 保存簇的奇异点
        """non_stable"""
    else:  # --------------------------------------竞争点
        if center[index_host] != center[index_nn] and not (0 < stable[index_nn] < 0.2):
            """注意：上面的条件：0<stable[index_nn]<0.2表示不稳定可以被聚合，但不具备扩展和被竞争的资格"""
            if degree_of_belonging(label_center, label_host, label_nn, node_dict, cluster_dict, cluster_nodes, cluster_label, sigmak2, sigmak2_inve, shares_num_max, knn, knn_nodes_large, knn_values_large, kesai_large) > 0:
                if not (label_nn in cluster_nodes_d[ccid]):
                    cluster_nodes_d[ccid].append(label_nn)
                if not ((label_host, label_host_root, altitude[index_host], label_nn, label_nn_root, outward_jd, outward_md) in cluster_belinked[ccid]):
                    cluster_belinked[ccid].append((label_host, label_host_root, altitude[index_host], label_nn, label_nn_root, outward_jd, outward_md))

    return center, cluster_nodes, cluster_nodes_d, vote_in, vote_out, edges, cluster_anomalies, cluster_belinked


def second_clustering(r, non_extendable, cluster_dict, ccenter, cluster_median, ctd_distance, cluster_clustering, cluster_nodes, edges_post1, edges_post2, cluster_inve_dict, node_dict, altitude):
    if not (r[0] in non_extendable or r[3] in non_extendable):
        c_host = cluster_dict[r[4]]
        c_nn = cluster_dict[r[1]]
        if ccenter[cluster_dict[r[1]]] != ccenter[cluster_dict[r[4]]]:
            """只要存在一个满足条件，即可进行合并的聚合"""
            if r[6] <= cluster_median[ccenter[c_host]] or r[6] <= cluster_median[ccenter[c_nn]]:
                th = cluster_dict[r[1]]
                ch = r[0]
                tn = cluster_dict[r[4]]
                cn = r[3]
                if ctd_distance[ccenter[th]] <= ctd_distance[ccenter[tn]]:
                    c_host = th
                    c_nn = tn
                    child_host = ch
                    child_nn = cn
                else:
                    c_host = tn
                    c_nn = th
                    child_host = cn
                    child_nn = ch

                cluster_clustering[ccenter[c_host]].extend(cluster_clustering[ccenter[c_nn]])  # 修改聚合盆点列表
                cluster_nodes[ccenter[c_host]].extend(cluster_nodes[ccenter[c_nn]])
                cluster_nodes[ccenter[c_nn]] = []

                edges_post1.append((child_host, child_nn))
                edges_post2.append((cluster_inve_dict[c_host], cluster_inve_dict[ccenter[c_nn]]))

                bak = copy.copy(cluster_clustering[ccenter[c_nn]])
                """修改原中心之后置为空"""
                cluster_clustering[ccenter[c_nn]] = []  # 修改为空列表
                for ctt in bak:
                    ccenter[cluster_dict[ctt]] = ccenter[c_host]
                cluster_median[ccenter[c_host]] = cluster_median_computing(cluster_nodes[ccenter[c_host]], altitude, node_dict)
    return ccenter, cluster_clustering, cluster_median, edges_post1, edges_post2, cluster_nodes



def Catchment_Basins_Clustering(data_embedding, data_size, node_dict, knn, knn_large, knn_nodes, knn_nodes_large, knn_values, knn_values_large, cluster_label, td_distance, kesai, kesai_large, sigmak):
    """＊＊＊＊＊＊＊＊------step1: 计算海拔值，和其他相关值--------＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊"""
    altitude, knn_altitude, knn_mean, knn_std, shares_num, shares_num_max, sigmak2, sigmak2_inve, sigma_hold, indegree, stable, non_stable_one, anomalies, non_stable_zero = data_preprocessing(data_size, node_dict, knn, knn_nodes, knn_large, knn_nodes_large, cluster_label,
                                                                                                                                                                                                td_distance, kesai, kesai_large, knn_values, knn_values_large, sigmak)

    """：：：：：：：：step2: 实现集水盆式聚类：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：："""
    node_and_altitude_ = list(zip(cluster_label, altitude))
    node_and_altitude = sorted(node_and_altitude_, key=lambda x: x[1], reverse=False)

    temp0 = list(zip([i for i in range(len(cluster_label))], node_and_altitude))  # 写入临时文件
    w = open('G_node_sorted.txt', 'w', encoding='utf8')
    for item in temp0:
        w.write(str(item) + '\n\n')
    w.close()
    del temp0  # 清除临时变量

    nodes_sorted = [item[0] for item in node_and_altitude]  # 临时保存排序后的节点列表

    vote_in = [list() for i in range(data_size)]  # 保存每个节点的入度节点
    vote_out = [list() for i in range(data_size)]  # 保存每个节点的出度节点

    center = [-1 for i in range(data_size)]  # 点的初始中心为-1

    cluster_dict = dict()  # 构建簇词典
    cluster_nodes = list()
    cluster_nodes_d = list()
    cluster_anomalies = list()
    cluster_belinked = list()

    edges = list()  # 有向边
    """－－－－－－－－－－－－－第1步集：集水盆的发现－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－"""

    for i in range(data_size):
        label_host = nodes_sorted[i]
        index_host = node_dict[nodes_sorted[i]]

        if 0 <= stable[index_host] < 0.2:  # 定义区间[0,2)为极不稳定性的节点
            continue

        if center[index_host] == -1:  # －－－－－－－－－－－－－－－－－－－－－出发点为盆点－－－－－－－－－
            center[index_host] = index_host

            cluster_dict[label_host] = len(cluster_dict)  # 增加一个簇

            cluster_nodes.append(list())
            cluster_nodes[cluster_dict[label_host]].append(label_host)  # 把簇中心点添加到簇
            cluster_nodes_d.append(list())
            cluster_nodes_d[cluster_dict[label_host]].append(label_host)  # 这里与cluster_nodes相同的内容

            cluster_anomalies.append(list())
            cluster_belinked.append(list())

            vote_in[index_host].append(label_host)

        label_center = cluster_label[center[index_host]]  # 节点的中心点的标签
        for j in range(knn_large):
            index_nn = node_dict[knn_nodes_large[index_host][j]]  # k近邻节点的词典序号
            label_nn = knn_nodes_large[index_host][j]  # k近邻节点的词典标签

            if label_nn in non_stable_zero:  # 0值不稳定点被视为 离群点
                continue

            if sigmak2[index_host][j] == 1 and sigmak2_inve[index_host][j] == 1 and shares_num_max[index_host][j] >= kesai_large:
                """注意：下面在聚合时，0<stable[index_nn]<0.2的不稳定可以被聚合，但不具备扩展和被竞争的资格"""
                center, cluster_nodes, cluster_nodes_d, vote_in, vote_out, edges, cluster_anomalies, cluster_belinked = first_clustering(index_host, index_nn, label_host, label_nn, center, altitude, cluster_label, label_center, cluster_dict, cluster_nodes,
                                                                                                                                         cluster_nodes_d, edges, vote_in, vote_out, anomalies, cluster_belinked, cluster_anomalies, node_dict, sigmak2,
                                                                                                                                         sigmak2_inve, shares_num_max, knn, knn_nodes, knn_nodes_large, knn_values_large, kesai_large, stable)
        '''－－－－－－－－－－－－－完成第1步集：建立集水盆－－－－－－－－－－－－－－－－－－－－－'''
    cluster_size = len(cluster_dict)  # －－－－－－词典大小
    cluster_inve_dict = list(cluster_dict.keys())  # －－－－－－逆向簇词典

    '''－－－－－－－－－－－－－－计算：每个簇的海拔中值－－－－－－－－－－－－－－－－－－－－－'''
    cluster_median = list()
    for x in range(cluster_size):
        cluster_median.append(cluster_median_computing(cluster_nodes[x], altitude, node_dict))

    """cluster_nodes_d介入子簇内各个点的簇内knn计算"""
    knn_in_cluster = cluster_os_distance(data_embedding, node_dict, cluster_nodes, cluster_nodes_d, cluster_size, knn, cluster_inve_dict, vote_in, indegree)

    """计算每个簇的点跨度和簇的平均跨度"""
    cp_span, norm_subc, span_nodes, non_extendable = pc_span(cluster_size, cluster_inve_dict, node_dict, cluster_nodes, vote_in, data_embedding, knn_in_cluster, knn_nodes, data_size, cluster_anomalies)

    '''－－－－－－－－－－－－－－进行：二次聚合－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－'''
    '''本算法在聚合过程中，从最小盆点出发，进行聚合，同一个大集水盆中，海拔越低的盆点很大可能先聚合'''
    ccenter = [i for i in range(cluster_size)]  # 记录簇之间形成的父子关系
    ctd_distance = [altitude[node_dict[x]] for x in cluster_inve_dict]  # 每个簇的中心点的海拔值
    cluster_clustering = [[cluster_inve_dict[i]] for i in range(cluster_size)]  # 保存聚合簇包含的簇
    edges_post1, edges_post2 = list(), list()  # 后期簇的聚合过程产生的连接边, edge_post1记录扩展关系产生的边，edge_post2记录扩展关系导致的盆点连接边

    connection = list()
    competition = list()
    for x in cluster_belinked:
        connection.extend(x)
        competition.extend([y[0] for y in x])  # －－－计算中心节点

    """－－－－－－－－－－－－－第2步集：集水盆的聚合－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－"""
    connection = sorted(connection, key=lambda x: x[2], reverse=False)
    for r in connection:
        """如果不存在不可扩展点，则直接进行聚合；否则，需要进行判断性的聚合"""
        if len(non_extendable) > 0:
            if r[0] in non_extendable or r[3] in non_extendable:
                continue
            else:
                ccenter, cluster_clustering, cluster_median, edges_post1, edges_post2, cluster_nodes = second_clustering(r, non_extendable, cluster_dict, ccenter, cluster_median, ctd_distance, cluster_clustering, cluster_nodes, edges_post1, edges_post2, cluster_inve_dict, node_dict, altitude)
        else:
            ccenter, cluster_clustering, cluster_median, edges_post1, edges_post2, cluster_nodes = second_clustering(r, non_extendable, cluster_dict, ccenter, cluster_median, ctd_distance, cluster_clustering, cluster_nodes, edges_post1, edges_post2, cluster_inve_dict, node_dict, altitude)

    cluster_center = list()
    for i in range(cluster_size):
        if len(cluster_nodes[i]) >= 1:
            cluster_center.append(cluster_nodes[i][0])
            for x in cluster_nodes[i]:
                center[node_dict[x]] = node_dict[cluster_inve_dict[i]]

    return center, vote_in, cluster_dict, edges, edges_post1, edges_post2, cluster_nodes, anomalies, competition, cluster_center, non_stable_zero, non_stable_one


def cluster_median_computing(single_cluster, altitude, node_dict):
    temp = [altitude[node_dict[x]] for x in single_cluster]
    median_node_altitude = np.median(np.array(temp))
    return median_node_altitude


def clustering(t, k, data_embedding):
    knn = t
    knn_large = k
    sigmak = 3
    kesai = int(knn / 2)
    kesai_large = int(knn_large / 2)

    cluster_label, pos, node_dict, data_size = read_dataset(data_embedding)

    '''***********计算每个点的基于欧氏距离的KNN, 平均距离, 均值偏差, 每个点的2度距离值（平均距离和均值偏差）**************************'''
    knn_nodes, knn_values, knn_nodes_large, knn_values_large = os_distance(data_size, data_embedding, cluster_label, knn, knn_large)

    td_distance, pos1 = point_td_distance(data_size, cluster_label, pos, knn_nodes, knn_values, node_dict)  # 用knn计算每个节点的共享距离的期望，而不是用knn_large
    """可视化朴素海拔"""
    """＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊"""
    center, vote_in, cluster_dict, edges, edges_post1, edges_post2, cluster_nodes, anomalies, competition, cluster_center, non_stable_zero, non_stable_one = Catchment_Basins_Clustering(data_embedding, data_size, node_dict, knn, knn_large,
                                                                                                                                                                                    knn_nodes, knn_nodes_large, knn_values, knn_values_large,
                                                                                                                                                                                    cluster_label, td_distance, kesai, kesai_large, sigmak)

    return center

