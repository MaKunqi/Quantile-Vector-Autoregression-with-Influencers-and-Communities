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
import os

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
    #alpha = (np.linalg.eigvalsh(Sigmahat)[-(num_clusters + 1)]) * np.sqrt(np.log(n)) / np.sqrt(T * np.min(P) ** 2)
    alpha=0.05
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
# 定义文件路径
file_path = 'norm_error.csv'

# 检查文件是否存在，如果不存在，则创建并写入标题
if not os.path.isfile(file_path):
    with open(file_path, 'w') as f:
        f.write('K,distance\n')  # 根据需要写入适当的标题
K_set=[5,10,15,20,25]
#K_set=[2,3,4,5]
N=100
results_df = pd.DataFrame(columns=['K', 'norm_error'])
for K in K_set:
    errors = []
    c_num = K
    c_size = int(N // c_num)
    r = N - c_num * c_size
    ind_star = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                         else int((i - r * (c_size + 1)) // c_size) + r for i in range(N)])
    z_star = get_index_matrix(c_num, ind_star,normalize=False)  # n*k
    v_star = np.zeros((c_num, N))  # k*n
    active_vals = [0.99, -5, 4, 6, -4]
    for j in range(c_num):
        v_star[j, j: j + 1] = np.array(active_vals[:1])
    # print(v_star)#k*n
    B_tuta = np.matmul(z_star, v_star)

    for _ in range(10):
        data=simulation(N,K)
        cluster_num = K  # 聚类数
        print(data)
        names = data.columns.values
        print(names)
        # data = data.iloc[:-18]
        data_filled = data.fillna(method='ffill')  # 向前向后对齐，使得没有缺失值
        data_filled = data_filled.fillna(method='bfill')
        data_filled = data_filled.fillna(method='bfill')
        # data_filled = data_filled.diff(periods=1)#对波动率数据差分
        _, n = np.shape(data)
        print(data_filled)
        zeronums = data_filled.isnull().sum()  # 统计每列空值的个数，并填充为0
        result_z, result_v, score = repeat_and_select_best(data_filled, n, cluster_num, repeat_times=10)
        theta_est = np.matmul(result_v, result_z)
        v_est = result_v
        z_est = result_z
        error=np.linalg.norm(B_tuta-theta_est.T, 'fro')/np.linalg.norm(B_tuta, 'fro')
        with open(file_path, 'a') as f:
            f.write(f'{K},{error}\n')
        errors.append(error)
    # 计算当前 K 值下的平均距离（期望聚类误差）
    expected_distance = np.mean(errors)
    # 将 K 值和计算出的平均距离添加到结果 DataFrame 中
    new_row = pd.DataFrame({'K': [K], 'norm_error': [expected_distance]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)
print(results_df)
# 使用 matplotlib 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(results_df['K'], results_df['norm_error'], marker='o')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Expected Norm Error')
plt.title('Expected Norm Error for different values of K')
plt.grid(True)
plt.savefig("simulation/expected norm error,tau0.5,k=5~25.png", dpi=600)
plt.show()
# 保存 DataFrame 到 CSV 文件
results_df.to_csv('norm_error_results.csv', index=False)
print('tau=',0.5)
print('K=5~25')
print('simulation data,tau=0.5')




