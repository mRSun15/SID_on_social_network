import numpy as np
import networkx.linalg
import networkx as nx
from scipy.linalg import *

def logistic_func(X, preference):
    reward = 0
    for i in range(len(preference)):
        reward += X[i]*preference[i]

    # return 1.0/(1+np.exp(-reward))
    return np.float(reward)

def estimate_reward(X, pref):
    expected_mean = pref['mean']

    return logistic_func(X, expected_mean)

def estimate_risk_adverse(X, est_friend_pref,rho = 0.2):
    estimate_mean = est_friend_pref['mean']
    estimate_cov = est_friend_pref['cov']

    risk_rewards = []
    sample_number = 100
    estimate_sample = np.random.multivariate_normal(mean=estimate_mean,
                                                       cov=estimate_cov,
                                                       size=sample_number)

    for i in range(sample_number):
        current_sample = estimate_sample[i]
        risk_rewards.append(logistic_func(X, current_sample))

    risk_rewards = np.array(risk_rewards)
    return np.var(risk_rewards)-rho*np.mean(risk_rewards)

def graph_eigen_centrality(Graph):
    adj_mat = nx.to_numpy_matrix(Graph)
    eigenvals, eigenvects = np.linalg.eig(adj_mat)
    idx = eigenvals.argsort()[::-1]
    eigenvects = eigenvects[:, idx]

    return eigenvects[:,0]


def graph_eigen(Graph):

    laplacian_mat = networkx.linalg.laplacian_matrix(Graph)
    l_mat = laplacian_mat.todense()
    l_inv = pinv(l_mat)

    eigenvals, eigenvects = np.linalg.eig(l_inv)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvects = eigenvects[:,idx]

    return eigenvals, eigenvects