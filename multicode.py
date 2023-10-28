import pandas as pd
import numpy as np
import pickle
from prepare import quantile_VAR,missing_var,random_basis,index_dist,get_index_matrix
from alternating import H,K,z_step,v_step,loss,z_new_step,v_new_step
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from cvxopt import spmatrix , sparse,solvers, matrix
from cvxopt.solvers import qp, lp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from statsmodels.regression.quantile_regression import QuantReg
def alternating(Y, alpha_v, V_start,epochs=200):#已知Y，V的开始值，聚类数，迭代计算
    T, n = Y.shape
    x, k = V_start.shape
    assert (x == n)
    assert (k <= n)
    matrix = np.zeros((k, n))
    for i in range(n):
        random_index = np.random.randint(0, k)
        matrix[random_index, i] = 1
    z_est=matrix
    v_est=V_start
    #print(matrix)
    for e in range(epochs):
        #print(e)
        v_est= v_new_step(Y,z_est,v_est)
        tmp = z_est.copy()
        z_est = z_new_step(Y,z_est, v_est)
        if np.array_equal(tmp, z_est):
            print(e)
            print("convergence")
            print('ohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            break
        else:
            print('no convergence')
            if e==199:
                print('ooooooooooooooooooooooooooooooooooooooops')
    score=loss(Y,v_est,z_est,alpha_v)

    return z_est, v_est, score

def quantile_sonic(data,n,num_clusters):
    V = random_basis(n, num_clusters)
    #print(V.shape)  # V是一个N*K的矩阵
    zeronums = data.isnull().sum()  # 统计每列空值的个数，并填充为0
    data = data.fillna(0)
    #print(data)
    T, n = np.shape(data)  # T是时间长度，n是个体个数,全部数据下n=28,T=1879
    P = np.zeros(n)  # P是pi的估计值构成的向量（矩阵形式）
    for i in range(n):
        P[i] = 1 - zeronums[i] / T
    Sigmahat = missing_var(data, P)  # 计算Sigmahat 28*28,协方差估计值
    Sigmahat = Sigmahat.values
    # 数据驱动找惩罚项系数
    alpha = (np.linalg.eigvalsh(Sigmahat)[-(num_clusters + 1)]) * np.sqrt(np.log(n)) / np.sqrt(T * np.min(P) ** 2)
    #print(alpha)
    ans=alternating(data, alpha,V)

    return ans


def repeat_and_select_best(data, n, cluster_num, repeat_times=10):
    best_result = None
    best_score = float('inf')  # 设置一个足够大的初值

    for _ in range(repeat_times):
        z_est, v_est, score = quantile_sonic(data, n, cluster_num)
        if score < best_score:  # 如果当前分数更低，则更新最佳结果
            best_score = score
            best_result = (z_est, v_est, score)

    return best_result

def compute_distance_for_cluster_num(cluster_num, data_filled):
    distance_cluster = np.zeros(5)
    data1 = data_filled.iloc[:109]
    data2 = data_filled.iloc[7:116]
    data3 = data_filled.iloc[14:123]
    data4 = data_filled.iloc[22:131]
    data5 = data_filled.iloc[29:138]
    data6 = data_filled.iloc[36:]
    result_z1, result_v1, _ = repeat_and_select_best(data1, n, cluster_num)
    indices1 = np.argmax(result_z1, axis=0)
    result_z2, result_v2, _ = repeat_and_select_best(data2, n, cluster_num)
    indices2 = np.argmax(result_z2, axis=0)
    result_z3, result_v3, _ = repeat_and_select_best(data3, n, cluster_num)
    indices3 = np.argmax(result_z3, axis=0)
    result_z4, result_v4, _ = repeat_and_select_best(data4, n, cluster_num)
    indices4 = np.argmax(result_z4, axis=0)
    result_z5, result_v5, _ = repeat_and_select_best(data5, n, cluster_num)
    indices5 = np.argmax(result_z5, axis=0)
    result_z6, result_v6, _ = repeat_and_select_best(data6, n, cluster_num)
    indices6 = np.argmax(result_z6, axis=0)
    distance_cluster[0] = index_dist(cluster_num, indices1, indices2)
    distance_cluster[1]= index_dist(cluster_num, indices3, indices1)
    distance_cluster[2] = index_dist(cluster_num, indices4, indices1)
    distance_cluster[3] = index_dist(cluster_num, indices5, indices1)
    distance_cluster[4]= index_dist(cluster_num, indices6, indices1)
    return cluster_num, distance


if __name__ == "__main__":
    # 读取和处理数据
    data = pd.read_csv('cleaned_data_new.csv')
    data = data.iloc[:-1]
    print(data)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    print(normalized_data)
    data = normalized_data
    data_filled = data.fillna(method='ffill')  # 向前向后对齐，使得没有缺失值
    data_filled = data_filled.fillna(method='bfill')
    data_filled = data_filled.fillna(method='bfill')
    distance = np.zeros((10, 5))
    _, n = np.shape(data)
    # 使用ThreadPoolExecutor进行多线程计算
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(compute_distance_for_cluster_num, i, data_filled) for i in range(2, 11, 1)]
        for future in concurrent.futures.as_completed(futures):
            cluster_num, dist = future.result()
            distance[(cluster_num - 2)] = dist
    print('tau=', 0.8)
    print('economy data, tau=0.8')
    print(distance)



