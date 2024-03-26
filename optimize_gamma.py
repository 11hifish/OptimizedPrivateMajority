from poibin.poibin import PoiBin
import numpy as np
from scipy.optimize import linprog
import time
from itertools import combinations_with_replacement
import scipy.special as ssp


## Method 1 of computing the objective: directly sample probabilities to compute the PB distribution
## take the empirical expectation as an approximaation to the true objective (expectation taken over all probs)
def compute_objective_coeff(p_vec):
    K = len(p_vec)
    mid = (K + 1) // 2
    rv = PoiBin(p_vec)
    prob_diff_vals = np.array([rv.pmf(l) - rv.pmf(K - l) for l in range(mid, K + 1)])
    obj_val_coeff = -0.5 * prob_diff_vals
    return obj_val_coeff

def estimate_expected_objective_coefficient_by_prob_samples(K, T=10000):
    # obj_val_est = 0
    all_obj_vals = []
    print('n trials: ', T)
    for trial in range(T):
        # sample probabilities
        p_vec = np.random.random(size=K)
        # get objective value based on sampled probabilities
        obj_val = compute_objective_coeff(p_vec)
        # obj_val_est += obj_val
        all_obj_vals.append(obj_val)
    all_obj_vals = np.array(all_obj_vals)
    obj_coeff_est = np.mean(all_obj_vals, axis=0)
    print(obj_coeff_est)
    print('estimated obj coeff: {}, std: {}'.format(obj_coeff_est, np.std(all_obj_vals, axis=0)))
    return obj_coeff_est


## Method 2 of computing the objetive: (unjustified)
## Approximate the integration over p_1, ..., p_K from 0.5 to 1.
def approximate_integration(K, p_lower=0.5, p_upper=1., sample_size=50, N=10000):
    mid = (K + 1) // 2
    l_range = np.arange(mid, K + 1)
    K_l_range = K - l_range
    res = np.zeros(len(l_range))
    p_samples = np.linspace(p_lower, p_upper, sample_size + 1)
    weight = ((p_samples[1] - p_samples[0]) ** K) * np.math.factorial(K)
    p_samples = p_samples[:-1]
    P = ssp.binom(sample_size + K - 1, K)
    for trial_idx in range(N):
        idx_chosen = np.random.choice(np.arange(sample_size), replace=True, size=K)
        probs = p_samples[idx_chosen]
        rv = PoiBin(probs)
        pmfs = rv.pmf(np.arange(K + 1))
        res += (pmfs[l_range] - pmfs[K_l_range]) * weight
    res = res * P / N  # form an unbiased estimation of the numerical integration
    return res


# note gamma is a vector of size (K + 1) // 2
def estimate_expected_objective_coefficient_by_approx_int(K, p_lower=0.5, p_upper=1, sample_size=30):
    int_p_vals = approximate_integration(K, p_lower, p_upper, sample_size=sample_size, N=10000)
    c = - int_p_vals
    return c


def get_constraints(K, m, eps, delta_0, delta):
    potential_pairs = [
                       np.array([0, 0]),
                       np.array([1, 1]),
                       np.array([delta_0, 0]), np.array([0, delta_0]),
                       np.array([(np.exp(eps) + delta_0) / (1 + np.exp(eps)), (1 - delta_0) / (1 + np.exp(eps))]),
                       np.array([(1 - delta_0) / (1 + np.exp(-eps)), (np.exp(eps) + delta_0) / (1 + np.exp(eps))]),
                       np.array([1, 1 - delta_0]), np.array([1 - delta_0, 1])
                       ]
    all_prob_pairs = list(combinations_with_replacement(potential_pairs, K))
    all_constr = []
    mid = (K + 1) // 2
    support = np.arange(0, K + 1)
    first_support = np.arange(0, mid)
    second_support = np.arange(mid, K + 1)
    print('# constraints: ', len(all_prob_pairs))
    for prob_pairs in all_prob_pairs:
        prob_pairs = np.array(prob_pairs)
        p_probs = prob_pairs[:, 0]
        p_prime_probs = prob_pairs[:, 1]
        rv_p = PoiBin(p_probs)
        pmf_p = rv_p.pmf(support)
        rv_p_prime = PoiBin(p_prime_probs)
        pmf_p_prime = rv_p_prime.pmf(support)
        coeff_first = np.exp(m * eps) * pmf_p_prime[first_support] - pmf_p[first_support]
        coeff_first_rev = coeff_first[::-1]
        coeff_second = pmf_p[second_support] - np.exp(m * eps) * pmf_p_prime[second_support]
        coeff_vec = coeff_first_rev + coeff_second
        all_constr.append(coeff_vec)
    all_constr = np.array(all_constr)
    target_ub = np.ones(len(all_prob_pairs)) * (np.exp(m * eps) - 1 + 2 * delta)
    return all_constr, target_ub


def set_constraints(model, var_list, K, m, eps, delta_0, delta):
    if delta_0 == 0 and delta == 0:
        print('reduced # constraints')
        potential_pairs = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([(np.exp(eps) + delta_0) / (1 + np.exp(eps)), (1 - delta_0) / (1 + np.exp(eps))]),
            np.array([(1 - delta_0) / (1 + np.exp(-eps)), (np.exp(eps) + delta_0) / (1 + np.exp(eps))]),
        ]
    else:
        potential_pairs = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([delta_0, 0]), np.array([0, delta_0]),
            np.array([(np.exp(eps) + delta_0) / (1 + np.exp(eps)), (1 - delta_0) / (1 + np.exp(eps))]),
            np.array([(1 - delta_0) / (1 + np.exp(-eps)), (np.exp(eps) + delta_0) / (1 + np.exp(eps))]),
            np.array([1, 1 - delta_0]), np.array([1 - delta_0, 1])
        ]
    all_prob_pairs = list(combinations_with_replacement(potential_pairs, K))
    mid = (K + 1) // 2
    support = np.arange(0, K + 1)
    first_support = np.arange(0, mid)
    second_support = np.arange(mid, K + 1)
    print('# constraints: ', len(all_prob_pairs))
    target_ub = np.exp(m * eps) - 1 + 2 * delta
    for constr_idx, prob_pairs in enumerate(all_prob_pairs):
        prob_pairs = np.array(prob_pairs)
        p_probs = prob_pairs[:, 0]
        p_prime_probs = prob_pairs[:, 1]
        rv_p = PoiBin(p_probs)
        pmf_p = rv_p.pmf(support)
        rv_p_prime = PoiBin(p_prime_probs)
        pmf_p_prime = rv_p_prime.pmf(support)
        coeff_first = np.exp(m * eps) * pmf_p_prime[first_support] - pmf_p[first_support]
        coeff_first_rev = coeff_first[::-1]
        coeff_second = pmf_p[second_support] - np.exp(m * eps) * pmf_p_prime[second_support]
        coeff_vec = coeff_first_rev + coeff_second
        # set model constraints
        model.addConstr(sum([var_list[i] * coeff_vec[i] for i in range(len(var_list))]) <= target_ub,
                        "constr_{}".format(constr_idx))
    return model


def optimize_gamma(K, eps, m, delta, delta_0):
    mid = (K + 1) // 2
    print('[PROBLEM] K = {}, m = {}, eps = {}, delta = {}, delta_0 = {}'.format(K, m, eps, delta, delta_0))
    # A, ub = get_constraints(K=K, m=m, eps=eps, delta_0=delta_0, delta=delta)
    A, ub = get_constraints(K=K, m=m, eps=eps, delta=delta, delta_0=delta_0)
    print(A.shape, ub.shape)
    bounds = [(0., 1.) for _ in range(mid)]
    c = estimate_expected_objective_coefficient_by_approx_int(K)  # Unjustified Method 2 of computing the objective [this one works]
    # c = estimate_expected_objective_coefficient_by_prob_samples(K)  # Method 1 of computing the objective [all 0 gamma]
    # c = estimate_expected_objective_coefficient_by_approx_int(K, p_lower=0, p_upper=1, sample_size=1000)  # [all 0 gamma]

    # define LP problem
    res = linprog(c, A_ub=A, b_ub=ub, bounds=bounds, method='highs-ipm')
    print(res.x)
    print(res.message)
    half_gamma = res.x
    optimized_gamma = np.concatenate((half_gamma[::-1], half_gamma))
    print('opt gamma: ', optimized_gamma)
    return optimized_gamma


def optimize_gamma_gurobi(K, eps, m, delta, delta_0, p_lower=0.5):
    import gurobipy as gp
    from gurobipy import GRB
    import matplotlib.pyplot as plt

    mid = (K + 1) // 2
    model = gp.Model("OptGamma")
    # create var list
    var_list = []
    for i in range(mid, K + 1):
        vi = model.addVar(name="p_{}".format(i))
        model.addConstr(vi <= 1, "prob_ub_{}".format(i))
        model.addConstr(vi >= 0, "prob_lb_{}".format(i))
        var_list.append(vi)
    c = estimate_expected_objective_coefficient_by_approx_int(K, p_lower=p_lower)
    model.setObjective(sum([var_list[i] * c[i] for i in range(len(var_list))]), GRB.MINIMIZE)
    # set privacy constraints
    set_constraints(model, var_list, K, m, eps, delta_0, delta)
    model.optimize()
    var_dic = {}
    for v in model.getVars():
        var_dic[v.varName] = v.x
        print(v.varName, v.x)
    half_gamma_fn = []  # upper half of gamma from mid to K
    for i in range(mid, K + 1):
        var_name = "p_{}".format(i)
        half_gamma_fn.append(var_dic[var_name])
    half_gamma_fn = np.array(half_gamma_fn)
    opt_gamma = np.concatenate((half_gamma_fn[::-1], half_gamma_fn))
    print('opt gamma: ', opt_gamma)
    return opt_gamma


def main():
    K = 11
    eps = 0.1
    mid = (K + 1) // 2
    m = 3
    # delta = 0.0001
    delta = 0
    delta_0 = delta / K

    opt_gamma = optimize_gamma(K, eps, m, delta, delta_0)
    # plot optimized gamma
    import matplotlib.pyplot as plt
    plt.plot(np.arange(K + 1), opt_gamma)
    plt.show()

def main_gurobi():
    import gurobipy as gp
    from gurobipy import GRB
    import matplotlib.pyplot as plt

    K = 11
    eps = 0.1
    m = 3
    mid = (K + 1) // 2
    # delta_0 = 0.001
    delta_0 = 0
    delta = 0
    # delta = delta_0 * m

    model = gp.Model("OptGamma")
    # create var list
    var_list = []
    for i in range(mid, K + 1):
        vi = model.addVar(name="p_{}".format(i))
        model.addConstr(vi <= 1, "prob_ub_{}".format(i))
        model.addConstr(vi >= 0, "prob_lb_{}".format(i))
        var_list.append(vi)
    c = estimate_expected_objective_coefficient_by_approx_int(K)
    model.setObjective(sum([var_list[i] * c[i] for i in range(len(var_list))]), GRB.MINIMIZE)
    # set privacy constraints
    set_constraints(model, var_list, K, m, eps, delta_0, delta)
    model.optimize()
    var_dic = {}
    for v in model.getVars():
        var_dic[v.varName] = v.x
        print(v.varName, v.x)
    half_gamma_fn = []  # upper half of gamma from mid to K
    for i in range(mid, K + 1):
        var_name = "p_{}".format(i)
        half_gamma_fn.append(var_dic[var_name])
    half_gamma_fn = np.array(half_gamma_fn)
    opt_gamma = np.concatenate((half_gamma_fn[::-1], half_gamma_fn))
    print('opt gamma: ', opt_gamma)
    plt.plot(np.arange(K + 1), opt_gamma)
    plt.show()



def test_itertool():
    delta_0 = 0.0001
    eps = 0.1
    K = 51
    potential_pairs = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([delta_0, 0]), np.array([0, delta_0]),
        np.array([(np.exp(eps) + delta_0) / (1 + np.exp(eps)), (1 - delta_0) / (1 + np.exp(eps))]),
        np.array([(1 - delta_0) / (1 + np.exp(-eps)), (np.exp(eps) + delta_0) / (1 + np.exp(eps))]),
        np.array([1, 1 - delta_0]), np.array([1 - delta_0, 1])
    ]
    t1 = time.time()
    all_prob_pairs = list(combinations_with_replacement(potential_pairs, K))
    t2 = time.time()
    print('time taks to generate all constraints for K = {}: {}'.format(K, t2 -t1))
    print(all_prob_pairs[11])
    print(np.array(all_prob_pairs[11]))


if __name__ == '__main__':
    # K = 11
    # m = 3
    # eps = 0.1
    # delta_0 = 0.01
    # get_constraints(K, m, eps, delta_0)

    # main()
    # estimate_expected_objective_coefficient(K=11,T=10000)
    # test_itertool()
    main_gurobi()
