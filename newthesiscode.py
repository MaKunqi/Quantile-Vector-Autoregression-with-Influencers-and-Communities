import pandas as pd
import numpy as np
import pickle
from prepare import quantile_VAR,missing_var,random_basis,index_dist,get_index_matrix
from alternating import H,K,z_step,v_step,loss,z_new_step,v_new_step
from simulation import simulation
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
        v_est= v_new_step(Y,alpha_v,z_est,v_est)
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
    print(alpha)
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


#N=100
distance=np.zeros((10,5))
data = pd.read_csv('RealData/macro40.csv',index_col=0) # read data
#data=data.iloc[:-1]
print(data)
#column_name_to_be_removed = 'Qatar'
#if column_name_to_be_removed in data.columns:
#    data = data.drop(columns=[column_name_to_be_removed])
normalized_data = (data - data.min()) / (data.max() - data.min())
print(normalized_data)
data=normalized_data
for i in range(2,3,1):
    cluster_num=i#聚类数
    #data = data.iloc[521:1565]
    '''
    data=simulation(N,cluster_num)
    c_size = int(N // cluster_num)
    r = N - cluster_num * c_size
    ind_star = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                                 else int((i - r * (c_size + 1)) // c_size) + r for i in range(N)])
    print(ind_star)
    '''

    print(data)
    names = data.columns.values
    print(names)
    #data = data.iloc[:-18]
    data_filled = data.fillna(method='ffill')#向前向后对齐，使得没有缺失值
    data_filled = data_filled.fillna(method='bfill')
    data_filled = data_filled.fillna(method='bfill')
    #data_filled = data_filled.diff(periods=1)#对波动率数据差分
    _, n = np.shape(data)
    print(data_filled)
    zeronums = data_filled.isnull().sum()  # 统计每列空值的个数，并填充为0
    #data_filled=data_filled.iloc[50:]
    #print(type(data_filled))
    #print(data_filled.shape)
    #fit = quantile_VAR(data_filled, configuration={'nlag': 1, 'tau': 0.5})
    #U, S, VT = np.linalg.svd(fit['B'])#计算得到开始值V
    #V = U[:, :cluster_num]

    result_z, result_v, score = repeat_and_select_best(data_filled, n, cluster_num,repeat_times=40)
    #with open('ecodata_first_no_lambda_in_loss,tau0.8,k=3.pickle', 'wb') as f:
    #   pickle.dump((result_z, result_v, score), f)
    theta_est=np.matmul(result_v,result_z)
    v_est=result_v
    z_est=result_z
    ind_est=np.argmax(result_z, axis=0)
    lists = [[] for i in range(cluster_num)]  # 按类将聚类元素分组
    for i in range(n):
        lists[ind_est[i]].append(i)
    print(lists)
    rearrange = []
    for j in range(cluster_num):
        rearrange += lists[j]
    print(rearrange)
    print(names)
    print(names[rearrange])
    theta_sort = theta_est[rearrange].T  # 重新按照聚类排列theta_est
    theta_sort = theta_sort[rearrange].T
    print(theta_sort)
    plt.figure()
    sns.set()
    ax = sns.heatmap(theta_sort, center=0, xticklabels=names[rearrange], yticklabels=names[rearrange], cmap="PiYG",
                     vmin=-0.1, vmax=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    plt.savefig("ecoheatmap/review_new_withoutlasso,tau0.8,k=2.png", dpi=600)
    plt.show()
    '''
    data1 = data_filled.iloc[:109]
    data2 = data_filled.iloc[7:116]
    data3 = data_filled.iloc[14:123]
    data4 = data_filled.iloc[22:131]
    data5 = data_filled.iloc[29:138]
    data6 = data_filled.iloc[36:]
    '''
    '''
    data1 = data_filled.iloc[:155]
    data2 = data_filled.iloc[10:165]
    data3 = data_filled.iloc[20:175]
    data4 = data_filled.iloc[30:185]
    data5 = data_filled.iloc[40:]
    '''
    '''
    #521-1043疫情前两年
    #1043-1565,2020-2021年疫情最严重的两年
    data1 = data_filled.iloc[521:1356]
    data2 = data_filled.iloc[573:1408]
    data3 = data_filled.iloc[625:1461]
    data4 = data_filled.iloc[678:1513]
    data5 = data_filled.iloc[730:1565]
    '''
    '''
    result_z1, result_v1, _ =repeat_and_select_best(data1,n,cluster_num)
    indices1 = np.argmax(result_z1, axis=0)
    result_z2, result_v2, _ =repeat_and_select_best(data2, n, cluster_num)
    indices2 = np.argmax(result_z2, axis=0)
    result_z3, result_v3, _ =repeat_and_select_best(data3, n, cluster_num)
    indices3 = np.argmax(result_z3, axis=0)
    result_z4, result_v4, _ =repeat_and_select_best(data4, n, cluster_num)
    indices4 = np.argmax(result_z4, axis=0)
    result_z5, result_v5, _ =repeat_and_select_best(data5, n, cluster_num)
    indices5 = np.argmax(result_z5, axis=0)
    result_z6, result_v6, _ = repeat_and_select_best(data6, n, cluster_num)
    indices6 = np.argmax(result_z6, axis=0)
    distance[(i - 2), 0] = index_dist(i, indices1, indices2)
    distance[(i - 2), 1] = index_dist(i, indices3, indices1)
    distance[(i - 2), 2] = index_dist(i, indices4, indices1)
    distance[(i - 2), 3] = index_dist(i, indices5, indices1)
    distance[(i - 2), 4] = index_dist(i, indices6, indices1)
    '''
    '''
    file_path = "D:\\work\\论文\\results_empirical_new_tau=0.5__K2~6.txt"
    # 使用'with'语句打开文件进行写入
    with open(file_path, 'a') as file:
        #file.write("Indices:\n")
        #file.write(str(indices))
        file.write("\n\nK:\n")
        file.write(str(cluster_num))
        file.write("\n\nResult:\n")
        file.write(str(result_z1))
        file.write(str(result_v1))
        file.write(str(result_z2))
        file.write(str(result_v2))
        file.write(str(result_z3))
        file.write(str(result_v3))
        file.write(str(result_z4))
        file.write(str(result_v4))
        file.write(str(result_z5))
        file.write(str(result_v5))
    print(f"Results saved to {file_path}")
    '''
print('tau=',0.8)
print('economy data,tau=0.5')
print(distance)



