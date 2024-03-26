from scipy.special import erfc
import numpy as np


def compute_qn(vote_histogram, sigma):
    n_classes = len(vote_histogram)
    q_vec = np.zeros(len(vote_histogram))
    for cls_i in range(n_classes):
        cls_count = vote_histogram[cls_i]
        diff_counts = cls_count - vote_histogram
        val = erfc(diff_counts / (2 * sigma))
        val[cls_i] = 0
        q_vec[cls_i] = 0.5 * np.sum(val)
    return q_vec

def compute_mu1_mu2_eps1_eps2(q, sigma):
    mu2 = sigma * np.sqrt(np.log(1 / q))
    mu1 = mu2 + 1
    eps1 = mu1 / (sigma ** 2)
    eps2 = mu2 / (sigma ** 2)
    return mu1, mu2, eps1, eps2

def compute_A(q, mu2, eps2):
    return (1 - q) / (1 - (q * np.exp(eps2)) ** ((mu2 - 1) / mu2))

def compute_B(q, mu1, eps1):
    return np.exp(eps1) / (q ** (1 / (mu1 - 1)))


def satisfies_condition(q, mu1, mu2, eps2, lbd):
    q_ub = np.exp((mu2 - 1) * eps2) / ((mu1 / (mu1 - 1) * mu2 / (mu2 - 1)) ** mu2)
    print('q bound satisfied? ', q <= q_ub)
    print('q: {:.4f}, q_ub: {:.4f}'.format(q, q_ub))
    print('q < 1? ', q < 1)
    print('mu1 >= lbd? ', mu1 >= lbd, mu1, lbd)
    print('mu2 > 1? ', mu2 > 1)
    return (q < 1) and (mu1 >= lbd) and (mu2 > 1) and (q <= q_ub)

def compute_data_dependent_privacy_bound(vote_histogram, sigma, lbd):
    q_vec = compute_qn(vote_histogram, sigma)
    all_bound_candidates = []
    for q in q_vec:
        mu1, mu2, eps1, eps2 = compute_mu1_mu2_eps1_eps2(q, sigma)
        if satisfies_condition(q, mu1, mu2, eps2, lbd):
            print('data-dependent bound is used! ')
            A = compute_A(q, mu2, eps2)
            B = compute_B(q, mu1, eps1)
            AB_term = (1 - q) * (A ** (lbd - 1)) + q * (B ** (lbd - 1))
            bound_candidate = 1 / (lbd - 1) * np.log(AB_term)
            all_bound_candidates.append(bound_candidate)
    data_indp_bound = lbd / (sigma ** 2)
    print('data indp bound: ', data_indp_bound)
    print('all bound candidates: ', all_bound_candidates)
    if len(all_bound_candidates) == 0:
        return data_indp_bound
    else:
        bound_candidate = min(all_bound_candidates)
        return min(bound_candidate, data_indp_bound)


if __name__ == "__main__":
    vote_histogram = np.array([11, 0])
    sigma = 12
    lbd = 34
    bound = compute_data_dependent_privacy_bound(vote_histogram, sigma, lbd)
    print('final bound: ', bound)
    # q_vec = compute_qn(vote_histogram, sigma)
    # print(q_vec)

