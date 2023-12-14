import numpy as np
import pandas as pd
from prepare import get_index_matrix
def simulation(N,c_num):
    c_size = int(N // c_num)
    r = N - c_num * c_size

    mean = np.zeros(N)  # 均值向量
    cov = np.eye(N)
    sample = np.random.multivariate_normal(mean, cov)
    sigma=np.ones(N)*0.5
    sigma=sigma.reshape(-1, 1)
    ind_star = np.array([int(i // (c_size + 1)) if i < r * (c_size + 1)
                             else int((i - r * (c_size + 1)) // c_size) + r for i in range(N)])
    z_star = get_index_matrix(c_num, ind_star,normalize=False)#n*k
    s=1
    #print(z_star)
    v_star = np.zeros((c_num, N))#k*n
    active_vals = [0.8, -5, 4, 6, -4]
    for j in range(c_num):
        v_star[j, j: j + s] = np.array(active_vals[:s])
    #print(v_star)#k*n
    B_tuta=np.matmul(z_star, v_star)
    print(B_tuta)
    ans=np.empty((1,N))
    for i in range(500):
        sigma=np.matmul(B_tuta,sigma)
        sample = np.random.multivariate_normal(mean, cov)
        sigma= sigma+sample.reshape(-1,1)
        ans=np.concatenate((ans, sigma.reshape(1,-1)), axis=0)
    #print(ans.shape)
    #print(ans)
    df = pd.DataFrame(ans)
    return df
#df.to_csv('simulation.csv', index=False)