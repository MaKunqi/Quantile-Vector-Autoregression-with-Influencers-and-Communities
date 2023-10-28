import pandas as pd
import numpy as np
from cvxopt import spmatrix , sparse,solvers, matrix
from cvxopt.solvers import qp, lp
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from scipy.linalg import sqrtm

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#already collect result of stability test, input by yourself
ans_idx=[[ 1,5,6,0], [ 14,21,3,4], [17,19,15,18], [ 26,26,16,16], [27,24,23,20]]


SIM_NUM=5
c_nums = [2, 3, 4, 5, 6]
plt.figure()
for i, c_num in enumerate(c_nums):
    plt.plot([c_num] * (SIM_NUM - 1), ans_idx[i], 'o', color='b')
    plt.xlabel("Different Number of Clusters: K")
plt.ylabel("Clustering results distance for different time periods")
plt.xticks(range(min(c_nums), max(c_nums) + 1))
plt.title('stability:economic_data_tau=0.5')
plt.savefig("stability/stability_economic_data_tau=0.5.png",dpi=600)
plt.show()


