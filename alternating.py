import pandas as pd
import numpy as np
from cvxopt import spmatrix , sparse,solvers, matrix
from cvxopt.solvers import qp, lp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.optimize import minimize
from conquer.linear_model import low_dim, high_dim, pADMM

h = 0.05
tau = 0.8
def H(v):
    if v<=-1:return 0
    if v>=1:return 1
    return 1/2+15/16*(v-2/3*v**3+1/5*v**5)
def H_derivation(v):
    if v<=-1:return 0
    if v>=1:return 0
    return 15/16*(1-2*v**2+v**4)
def K(x):
    global h,tau
    return x*(H(x/h)+tau-1)

def quantile_loss(u):
    global tau
    return 0.5 * np.abs(u) + (tau - 0.5) * u
def loss(Y,v,z,alpha):
    T, n = Y.shape
    x, k = v.shape
    p, q = z.shape
    assert (x == n)
    assert (k <= n)
    assert (k == p)
    assert (q == n)
    sum = 0
    for t in range(T - 1):
        for i in range(n):
            #sum += K(x=Y.iloc[t + 1,i] - np.matmul(np.matmul(v, z[:, i]).T, Y.iloc[t].T)) + alpha * np.linalg.norm(z[:, i], ord=1)
            sum += K(x=Y.iloc[t + 1, i] - np.matmul(np.matmul(v, z[:, i]).T, Y.iloc[t].T))
    print(sum)
    return sum

def v_step(Y_in, lamb_in, z_in,v_initial): #lamb_in是惩罚项系数，v_initial是开始位置，梯度下降求新v=n*k维矩阵
    def reshape_to_matrix(v_flat, shape):
        return v_flat.reshape(shape)
    def create_loss_function(Y, z, lamb, v_shape):#计算损失函数
        def loss_function_with_fixed_params(v_flat):#v_flat是v的展开成向量形式
            v = reshape_to_matrix(v_flat, v_shape)  # 将v_flat转回矩阵形式
            T, n = Y.shape#T*n
            x, k = v.shape#n*k
            p, q = z.shape#k*n
            assert (x == n)
            assert (k <= n)
            assert (k == p)
            assert (q == n)
            sum = 0
            for t in range(T - 1):
                for i in range(n):
                    sum += K(x=Y.iloc[t + 1,i] - np.matmul(np.matmul(v, z[:, i]).T, Y.iloc[t].values.reshape(-1, 1))
                             ) + lamb * np.linalg.norm(z[:, i], ord=1)
            return sum

        return loss_function_with_fixed_params

    def create_grad_function(Y, z, lamb, v_shape):#计算损失函数
        def grad_function_with_fixed_params(v_flat):#v_flat是v的展开成向量形式
            v = reshape_to_matrix(v_flat, v_shape)  # 将v_flat转回矩阵形式
            T, n = Y.shape
            x, k = v.shape
            p, q = z.shape
            assert (x == n)
            assert (k <= n)
            assert (k == p)
            assert (q == n)
            sum = 0
            for t in range(T - 1):
                for i in range(n):#h=10
                    sum += H_derivation((Y.iloc[t+1,i]-np.matmul(np.matmul(z[:, i].reshape(1, -1),v.T),Y.iloc[t].values.reshape(-1, 1)))/10)*(Y.iloc[t+1,i]-np.matmul(np.matmul(z[:, i].reshape(1, -1),v.T),Y.iloc[t].values.reshape(-1, 1)))*(-np.matmul(Y.iloc[t].values.reshape(-1, 1),z[:, i].reshape(1, -1)))/10 - \
                           np.matmul(Y.iloc[t].values.reshape(-1, 1),z[:, i].reshape(1, -1))*H((Y.iloc[t+1,i]-np.matmul(np.matmul(z[:, i].reshape(1, -1),v.T),Y.iloc[t].values.reshape(-1, 1)))/10)
            sum +=lamb * np.sign(v)
            sum=sum.flatten()
            return sum

        return grad_function_with_fixed_params

    Y, z, lamb = Y_in,z_in,lamb_in # 你的已知参数
    v_initial_matrix = v_initial  # 你的初始矩阵v的值
    x0 = v_initial_matrix.flatten()  # 将矩阵v展平为一个向量
    v_shape = v_initial_matrix.shape  # 记录v的形状以供后续使用
    objective_function = create_loss_function(Y, z, lamb, v_shape)
    gradient_function=create_grad_function(Y, z, lamb, v_shape)
    print('v_step')
    res = minimize(objective_function, x0, method='Newton-CG', jac=gradient_function,
                   options={'maxiter': 100})
    tmp=res.x
    return reshape_to_matrix(tmp, v_shape)#返回v,形状是n*k

def v_new_step(Y,lamb_in, z,v): #lamb_in是惩罚项系数，v_initial是开始位置，梯度下降求新v=n*k维矩阵
    global h, tau
    print('v_new_step')
    T, n = Y.shape#T*n
    x, k = v.shape#n*k
    p, q = z.shape#k*n
    assert (x == n)
    assert (k <= n)
    assert (k == p)
    assert (q == n)
    ans=np.zeros((n,1))
    for i in range(k):
        print(i)
        N= np.sum(z[i, :])#即Ni,第i个聚类的元素个数
        N=int(N)
        if N==0:
            beta_value=np.zeros((1,n))
            beta_value_reshaped = beta_value.reshape(-1, 1)
            # print(beta_value.shape)
            ans = np.hstack((ans, beta_value_reshaped))
            continue
        indices_of_ones = np.where(z[i, :]== 1)[0]
        '''
        Y_tuta=np.zeros((1,1))
        for index in indices_of_ones:
            for t in range(T-1):
                Y_tuta=  np.vstack([Y_tuta, Y.iloc[t+1,index]])
        Y_tuta=Y_tuta[1:,:]
        print(Y_tuta)
        '''
        Y_tuta = Y.iloc[1:, indices_of_ones].values.flatten(order='F').reshape(-1, 1)
        '''
        X=np.zeros((1,n))
        for _ in range(N):
            for t in range(T-1):
                X =  np.vstack([X, Y.iloc[t, :]])
        X=X[1:,:]
        print(X)
        '''
        X = np.tile(Y.iloc[:-1, :].values, (N, 1))
        sqr = high_dim(X, Y_tuta, intercept=True)
        #lambd= np.quantile(sqr.self_tuning(tau),0.95)
        lambd=lamb_in
        ## l1-penalized conquer
        l1_model = sqr.l1(tau=tau, Lambda=lambd)
        beta_value = l1_model['beta']
        beta_value_reshaped = beta_value.reshape(-1, 1)
        #print(beta_value_reshaped)
        #print(beta_value.shape)
        ans = np.hstack((ans, beta_value_reshaped[1:]))
    ans=ans[:,1:]
    return ans



def z_step(Y, v):#已知v,枚举寻找最好的z
    print('z_step')
    T, n = Y.shape
    x, k = v.shape
    ans = np.empty((k, 0))
    assert (x == n)
    assert (k <= n)

    for j in range(n):
        tmp = []
        for i in range(k):
            sum = 0
            vector = np.zeros(k)
            vector[i] = 1
            for t in range(T - 1):
                sum += quantile_loss(Y.iloc[t + 1, i] - np.matmul(np.matmul(v, vector).T, Y.iloc[t, :]))
            tmp.append(sum)

        min_value = min(tmp)
        min_index = tmp.index(min_value)
        vector = np.zeros(k)
        vector[min_index] = 1
        ans = np.column_stack((ans, vector))

    print(ans.shape)
    return ans

def z_new_step(Y,z, v):#已知v,枚举寻找最好的一步改变的z
    print('z_new_step')
    T, n = Y.shape
    x, k = v.shape
    p, q = z.shape
    assert (x == n)
    assert (k <= n)
    assert (k == p)
    assert (q == n)
    index=[0,0]
    tmp=10000
    for j in range(n):
        for i in range(k):
            vector = np.zeros((k, 1))
            vector[i] = 1

            # 使用np.matmul进行矩阵运算，并利用numpy广播进行向量与矩阵的减法
            sum_val = np.sum(quantile_loss(
                Y.iloc[1:, j].values - np.matmul(np.matmul(v, vector).T, Y.iloc[:-1, :].values.T).flatten()) -
                             quantile_loss(Y.iloc[1:, j].values - np.matmul(np.matmul(v, z[:, j]).T,
                                                                            Y.iloc[:-1, :].values.T).flatten()))

            if sum_val < tmp:
                tmp = sum_val
                index = [j, i]

    z[:, index[0]] = 0  # 将第j列的所有元素设置为0
    z[index[1], index[0]] = 1
    return z