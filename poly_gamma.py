import numpy as np
import scipy.special as ssp
from scipy.stats import hypergeom


def h_fn_single(l, K, i, n_prime):
    return ssp.binom(l, i) * ssp.binom(K - l, n_prime - i) / ssp.binom(K, n_prime)
def h_fn(l, K, n_prime):
    mid = (n_prime + 1) // 2
    i_support = np.arange(mid, n_prime + 1)
    return sum([h_fn_single(l, K, i, n_prime) for i in i_support])

def baseline_h(l, m, K, c=1):
    mid = (m + 1) // 2
    i_support = np.arange(mid, m+1)
    rv = hypergeom(M=K, n=l, N=m)
    pmfs = c * rv.pmf(i_support)
    return sum(pmfs)

def twom_h(l, m, K):
    i_support = np.arange(m, 2 * m)
    rv = hypergeom(M=K, n=l, N=2*m-1)
    pmfs = rv.pmf(i_support)
    return sum(pmfs)

def gamma_2m(l, m, K):
    mid = (K + 1) // 2
    if l < mid:
        return 1 - 2 * twom_h(l, m, K)
    else:
        return 2 * twom_h(l, m, K) - 1

def subsampling_h(l, m, K):
    if m % 2 == 1:  # odd m
        mid = (m + 1) // 2
    else:  # even m
        mid = m // 2 + 1
    i_support = np.arange(mid, m + 1)
    rv = hypergeom(M=K, n=l, N=m)
    pmfs = rv.pmf(i_support)
    s = sum(pmfs)
    if m % 2 == 0:
        s += 0.5 * rv.pmf(m // 2)
    return s


def gamma_subsampling(l, m, K):
    mid = (K + 1) // 2
    if l < mid:
        return 1 - 2 * subsampling_h(l, m, K)
    else:
        return 2 * subsampling_h(l, m, K) - 1

def baseline_2m_gamma(K, m):
    if m >= (K + 1) / 2:
        return np.ones(K + 1)
    else:
        return np.array([gamma_2m(l, m, K) for l in range(0, K + 1)])

def baseline_gamma(K, m):
    return np.array([gamma_subsampling(l, m, K) for l in range(0, K + 1)])


# Other gamma baselines
def get_data_independent_gamma(K, m, eps):
    v1 = np.exp(m * eps) - 1
    v2 = 2 * (np.exp(K * eps) - np.exp(m * eps)) / (1 + np.exp(K * eps)) + v1
    return v1 / v2 * np.ones(K + 1)


def get_ideal_data_independent_gamma(K, m, eps, delta_star):
    thres = (1 / K) * np.log((np.exp(K * eps) + 1) / 2)
    if delta_star ** 2 > thres:
        print('Ideal Gamma activated !!!')
        v1 = np.exp(m * eps) - 1
        v2 = (1 - np.exp(m * eps - K * eps)) * np.exp(-K * (delta_star ** 2)) + v1
        return v1 / v2
    else:
        return get_data_independent_gamma(K, m, eps)


def RR_deta_independent_gamma(K, tau_eps, m, eps, lbda, delta):
    # aggregating K (eps, delta_0)-DP teachers is (tau_eps, lbda)-DP
    # we want the final output to be (m eps, delta)-DP
    print('K = {}, tau_eps = {}, m= {}, eps = {}, lbda = {}, delta = {}'.format(K, tau_eps, m, eps, lbda, delta))
    gamma_upper = np.exp(m * eps) - 1 + 2 * delta
    gamma_lower = 2 * (np.exp(tau_eps) - np.exp(m * eps) + (1 + np.exp(m * eps)) * lbda) / (np.exp(tau_eps) + 1) \
                  + np.exp(m * eps) - 1
    gamma = gamma_upper / gamma_lower
    return np.ones(K + 1) * gamma

