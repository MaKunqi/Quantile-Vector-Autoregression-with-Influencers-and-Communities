import pandas as pd
import numpy as np
from cvxopt import spmatrix , sparse,solvers, matrix
from cvxopt.solvers import qp, lp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from statsmodels.regression.quantile_regression import QuantReg
from conquer.linear_model import low_dim, high_dim, pADMM


def random_basis(n, k):
    assert (n >= k)
    x = np.random.randn(n, k)
    return np.matmul(x, sqrtm(np.linalg.inv(np.matmul(x.T, x))))

def missing_var(X, P): #X是数据(T*n)，P是pi的估计值构成的向量（矩阵形式），返回协方差阵估计值
    T = np.shape(X)[0]  #数据的时间长度
    S = X.T @ X
    S = np.matmul(X.T, X) / T  #观测到的经验协方差阵
    print(S.shape)
    D = np.diag(np.diag(S))
    Delta = np.diag(P ** (-1))
    Sigma = np.matmul(Delta, np.matmul(S - D, Delta)) + np.matmul(Delta, D)
    return Sigma

def missing_cross_var(X, P): #X是数据(T*n)，P是pi的估计值构成的向量（矩阵形式），返回交叉协方差阵估计值
    T = np.shape(X)[0]  #数据的时间长度
    X1 = X.iloc[:-1, :]  #X1,X2均是数据，X2滞后1阶
    X2 = X.iloc[1:, :]
    X1_np = X1.to_numpy()
    X2_np = X2.to_numpy()
    S = np.matmul(X1_np.T, X2_np) / (T-1)  #观测到的经验交叉协方差阵
    Delta = np.diag(P ** (-1))
    A = np.matmul(Delta, np.matmul(S, Delta))
    return A

def get_index_matrix(cluster_num, ind, normalize=True):  #已知聚类ind,聚类数，求索引矩阵z n*k
    n = np.size(ind)
    ind_mat = np.zeros((n, cluster_num))

    for i in range(n):
        ind_mat[i, ind[i]] = 1.

    if normalize:  #标准化索引矩阵z
        for j in range(cluster_num):
            x = ind_mat[:, j]
            norm = np.linalg.norm(x)
            if norm > 0.000000001:
                ind_mat[:, j] = x / norm
            else:
                # print("WARNING!!!")
                pass

    return ind_mat

def index_dist(cluster_num, int1, int2):  #求两个聚类的距离，int1,int2分别代表两个聚类
    n = np.size(int1)
    assert (np.size(int2 == n))

    b = np.matmul(get_index_matrix(cluster_num, int1, normalize=False).T,
                  get_index_matrix(cluster_num, int2, normalize=False))
    b = np.reshape(b, (cluster_num * cluster_num,))

    I = spmatrix(1.0, range(cluster_num ** 2), range(cluster_num ** 2))
    row_ones = spmatrix(
        1.0,
        sum([[j] * cluster_num for j in range(cluster_num)], []),
        sum([[cluster_num * j + k for k in range(cluster_num)] for j in range(cluster_num)], [])
    )
    col_ones = spmatrix(
        1.0,
        sum([[j] * cluster_num for j in range(cluster_num)], []),
        sum([[cluster_num * k + j for k in range(cluster_num)] for j in range(cluster_num)], []),
    )

    res = lp(-matrix(b),
             -I, matrix(0.0, size=(cluster_num ** 2, 1)),
             sparse([row_ones, col_ones])[:-1, :], matrix(1.0, size=(2 * cluster_num - 1, 1)))

    return n + round(res['primal objective'])

def quantile_VAR(x, configuration={'nlag': 1, 'tau': 0.5}):#计算最初的出发项Theta
    tau = float(configuration['tau'])
    nlag = int(configuration['nlag'])

    if not isinstance(x, pd.DataFrame):
        raise ValueError("Data needs to be of type 'pandas DataFrame'")

    if nlag <= 0:
        raise ValueError("nlag needs to be a positive integer")

    if tau <= 0 or tau >= 1:
        raise ValueError("tau needs to be within 0 and 1")

    k = x.shape[1]
    names = x.columns.tolist()
    if not names:
        names = list(range(1, k + 1))

    res = []
    B = []

    for i in range(k):
        # Embedding the data
        z = pd.concat([x.shift(j) for j in range(nlag + 1)], axis=1).dropna()
        y = z.iloc[:, i]
        X = z.iloc[:,k:]
        X.loc[:, 'Intercept'] = 1.0

        # Fitting Quantile regression
        model = QuantReg(y, X)
        fit = model.fit(q=tau)

        # Exclude the intercept
        slope_coeff = fit.params.values[:-1]
        B.append(slope_coeff)
        res.append(fit.resid.values)

    B = np.vstack(B)
    Res = np.column_stack(res)
    Q = np.dot(Res.T, Res) / Res.shape[0]
    Q = Q[np.newaxis, ...]

    results = {'B': B, 'Q': Q}
    #print(z)
    return results


def quantile_VAR_lasso(x, configuration={'nlag': 1, 'tau': 0.5}):#计算最初的出发项Theta
    tau = float(configuration['tau'])
    nlag = int(configuration['nlag'])

    if not isinstance(x, pd.DataFrame):
        raise ValueError("Data needs to be of type 'pandas DataFrame'")

    if nlag <= 0:
        raise ValueError("nlag needs to be a positive integer")

    if tau <= 0 or tau >= 1:
        raise ValueError("tau needs to be within 0 and 1")

    k = x.shape[1]
    names = x.columns.tolist()
    if not names:
        names = list(range(1, k + 1))

    res = []
    B = []

    for i in range(k):
        # Embedding the data
        z = pd.concat([x.shift(j) for j in range(nlag + 1)], axis=1).dropna()
        y = z.iloc[:, i]
        X = z.iloc[:,k:]
        # X['Intercept'] = 1.0

        # Fitting Quantile regression with lasso
        sqr = high_dim(X, y.to_numpy(), intercept=True)
        lambd = np.quantile(sqr.self_tuning(tau), 0.95)
        ## l1-penalized conquer
        l1_model = sqr.l1(tau=tau, Lambda=lambd)
        beta_value = l1_model['beta']
        beta_value_reshaped = beta_value.reshape(-1, 1)

        # Exclude the intercept
        B.append(beta_value[1:])
        # res.append(fit.resid.values)

    B = np.vstack(B)
    # Res = np.column_stack(res)
    # Q = np.dot(Res.T, Res) / Res.shape[0]
    # Q = Q[np.newaxis, ...]

    results = {'B': B, 'Q': 1}
    #print(z)
    return results