from optimize_gamma import optimize_gamma, optimize_gamma_gurobi
from poly_gamma import baseline_gamma, RR_deta_independent_gamma, baseline_2m_gamma
from utility import compute_estimated_TV_distance, compute_composed_privacy_and_failure_prob_by_subsamples
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

legend_labels = {
    'optimized': "$\gamma_{opt}$ (Ours)",
    'subsampling': "$\gamma_{Sub}$ (Baseline)",
    'data_indp': "$\gamma_{const}$ (Baseline)",
    'double_subsampling': "$\gamma_{DSub}$ (Baseline)"
}

def plot_gamma(list_gamma_functions, gamma_function_names):
    assert (len(list_gamma_functions) == len(gamma_function_names))
    for gamma_fn, name in zip(list_gamma_functions, gamma_function_names):
        plt.plot(np.arange(len(gamma_fn)), gamma_fn, label=name)
    plt.legend()
    plt.grid()
    plt.title('Gamma Function')
    plt.xlabel('K')
    plt.show()

def plot_estimated_TV_distance(list_gamma_functions, gamma_function_names, no_exp=10):
    assert (len(list_gamma_functions) == len(gamma_function_names))
    all_results = np.zeros((len(list_gamma_functions), no_exp))
    for exp_idx in range(no_exp):
        for fn_idx in range(len(list_gamma_functions)):
            gamma_fn = list_gamma_functions[fn_idx]
            est_TV_distance = compute_estimated_TV_distance(gamma_fn)
            all_results[fn_idx, exp_idx] = est_TV_distance
    for fn_idx in range(len(list_gamma_functions)):
        mean_est_TV_dist = np.mean(all_results[fn_idx])
        std_est_TV_dist = np.std(all_results[fn_idx])
        print(gamma_function_names[fn_idx], mean_est_TV_dist, std_est_TV_dist)
        plt.scatter(fn_idx, mean_est_TV_dist)
        plt.errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, label=gamma_function_names[fn_idx])
    plt.legend()
    plt.grid()
    plt.title('Error')
    plt.show()


def main_simple():
    K = 101
    eps = 0.1
    # K = 11
    # eps = 0.03
    # m = 5
    # all_ms = [1, 3, 5, 7, 9, 11]
    all_ms = [10, 20, 30, 40, 60, 80]
    delta_0 = 0
    # delta_0 = 1e-5
    all_opt_gammas = []
    all_subsample_gammas = []
    all_double_subsample_gammas = []
    all_indp_gamma = []
    for m in all_ms:
        delta = delta_0 * m

        opt_gamma_path = 'opt_gamma_K_{}_eps_{}_m_{}_pure_DP.pkl'.format(K, eps, m)
        if delta_0 == 0 and os.path.isfile(opt_gamma_path):
            with open(opt_gamma_path, 'rb') as f:
                opt_gamma = pickle.load(f)
            print('Loading pre-optimized opt gamma from {} ...'.format(opt_gamma_path))
        else:
            print('Optimizing gamma K = {}, eps = {}, m ={}, delta = {}, delta_0 = {}'.format(K, eps, m, delta, delta_0))
            opt_gamma = optimize_gamma_gurobi(K, eps, m, delta, delta_0)
            with open(opt_gamma_path, 'wb') as f:
                pickle.dump(opt_gamma, f)

        all_opt_gammas.append(opt_gamma)

        subsample_gamma = baseline_gamma(K, m)
        all_subsample_gammas.append(subsample_gamma)

        double_subsample_gamma = baseline_2m_gamma(K, m)
        all_double_subsample_gammas.append(double_subsample_gamma)

        data_indp_gamma = RR_deta_independent_gamma(K=K, tau_eps=K * eps, m=m, eps=eps, lbda=K * delta_0, delta=delta)
        all_indp_gamma.append(data_indp_gamma)
    # plot everything together
    color_map = {
        'optimized': 'r',
        'subsampling': 'b',
        'data_indp': 'g',
        'double_subsampling': 'cyan'
    }
    # legend_labels = {
    #     'optimized': 'Optimized DaRRM$_{\gamma}$ (Ours)',
    #     'subsampling': 'Subsampling (Baseline)',
    #     'data_indp': 'Randomized Response (Baseline)'
    # }


    get_legend = False
    if get_legend:
        fig, axes = plt.subplots(3, 2, figsize=(40, 40))
        fontsize = 60
    else:
        # fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig, axes = plt.subplots(3, 2, figsize=(10, 9))
        fontsize = 20
    for i in range(len(all_ms)):
        row_idx = i // 2
        col_idx = i % 2
        alpha = 0.5
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_opt_gammas[i], c=color_map['optimized'],
                                    label=legend_labels['optimized'], alpha=alpha)
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_subsample_gammas[i], c=color_map['subsampling'],
                                    label=legend_labels['subsampling'], alpha=alpha)
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_double_subsample_gammas[i], c=color_map['double_subsampling'],
                                    label=legend_labels['double_subsampling'], alpha=alpha)
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_indp_gamma[i], c=color_map['data_indp'],
                                    label=legend_labels['data_indp'], alpha=alpha)
        axes[row_idx, col_idx].set_title('m = {}'.format(all_ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].set_xticks([0, K / 2, K], [0, K / 2, K], fontsize=fontsize)
        axes[row_idx, col_idx].set_yticks([0,0.5, 1], [0, 0.5, 1], fontsize=fontsize)
        if not get_legend:
            axes[row_idx, col_idx].grid()
        if get_legend and i == 3:
            legend_fontsize = 15
            legend = axes[row_idx, col_idx].legend(fontsize=legend_fontsize, ncol=4, loc='lower right')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.text(0.5, 0.01, 'Support $l \in \{0,1,...,K\}$', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, '$\gamma$ values', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('Shape of $\gamma$ functions', fontsize=fontsize)

    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    plt.savefig('gamma_fn_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()

    def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
        for legobj in legend.legendHandles:
            legobj.set_linewidth(2)
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        plt.close()

    if get_legend:
        export_legend(legend, filename='legend_new.png')
        return


    no_exp = 5
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    ## plot estimated TV distance
    for i in range(len(all_ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=fontsize)
        axes[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=fontsize)
        # compute estimated TV distance
        list_gamma_functions = [all_opt_gammas[i], all_subsample_gammas[i], all_double_subsample_gammas[i], all_indp_gamma[i]]
        color_map_2 = [color_map['optimized'], color_map['subsampling'], color_map['double_subsampling'], color_map['data_indp']]
        all_results = np.zeros((len(list_gamma_functions), no_exp))
        for exp_idx in range(no_exp):
            for fn_idx in range(len(list_gamma_functions)):
                gamma_fn = list_gamma_functions[fn_idx]
                est_TV_distance = compute_estimated_TV_distance(gamma_fn)
                if est_TV_distance < 1e-5:
                    est_TV_distance = 0
                all_results[fn_idx, exp_idx] = est_TV_distance
        for fn_idx in range(len(list_gamma_functions)):
            mean_est_TV_dist = np.mean(all_results[fn_idx])
            std_est_TV_dist = np.std(all_results[fn_idx])
            # plot avg error of a specific funtion
            axes[row_idx, col_idx].scatter(fn_idx, mean_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_title('m = {}'.format(all_ms[i]), fontsize=fontsize)
        if not get_legend:
            axes[row_idx, col_idx].grid()
    # fig.text(0.5, 0.01, 'Error', ha='center', fontsize=fontsize)
    # fig.text(0.04, 0.5, '$\gamma$ functions', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.01, '$\gamma$ functions', ha='center', fontsize=fontsize)
    fig.text(0.01, 0.5, 'Error', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('$\mathcal{E}$(DaRRM$_{\gamma})$', fontsize=fontsize)
    # plt.show()
    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    plt.savefig('error_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()

    ## old version of plotting scripts
    # plot_gamma([opt_gamma, subsample_gamma], ['optimized', 'simple subsampling'])

    # plot estimated TV distance
    # plot_estimated_TV_distance([opt_gamma, subsample_gamma], ['optimized', 'simple subsampling'])


def main_advanced():
    def compute_m(k, delta_prime, eps):
        return np.sqrt(2 * k * np.log(1 / delta_prime)) + k * (np.exp(eps) - 1)

    K = 35
    eps = 0.1
    all_Ms = [10, 13, 15, 20]
    delta_0 = 1e-5
    delta_prime = 0.1
    all_opt_gammas = []
    all_subsample_gammas = []
    all_indp_gamma = []
    for M in all_Ms:
        delta = M * delta_0 + delta_prime

        if M == 10:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_7.83784960517159_pure_DP.pkl'
        elif M == 13:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_9.104612478173365_pure_DP.pkl'
        elif M == 15:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_9.888854452480267_pure_DP.pkl'
        elif M == 20:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_11.700470185889117_pure_DP.pkl'
        else:
            raise Exception('Unsupported subsampling size M = {}!'.format(M))

        with open(opt_gamma_path, 'rb') as f:
            opt_gamma = pickle.load(f)
        all_opt_gammas.append(opt_gamma)

        subsample_gamma = baseline_gamma(K, M)
        all_subsample_gammas.append(subsample_gamma)

        # compute parameters for data-independent gamma
        tau_eps = np.sqrt(2 * K * np.log(1 / delta_prime)) * eps + K * eps * (np.exp(eps) - 1)
        lbda = K * delta_0 + delta_prime
        m = compute_m(M, delta_prime, eps)
        data_indp_gamma = RR_deta_independent_gamma(K=K, tau_eps=tau_eps, m=m, eps=eps, lbda=lbda, delta=delta)
        all_indp_gamma.append(data_indp_gamma)
    # plot everything together
    color_map = {
        'optimized': 'r',
        'subsampling': 'b',
        'data_indp': 'g'
    }
    # legend_labels = {
    #     'optimized': 'Optimized DaRRM$_{\gamma}$ (Ours)',
    #     'subsampling': 'Subsampling (Baseline)',
    #     'data_indp': 'Randomized Response (Baseline)'
    # }
    fontsize = 20
    get_legend = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # fig, axes = plt.subplots(2, 2, figsize=(40, 40))
    for i in range(len(all_Ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_opt_gammas[i], c=color_map['optimized'],
                                    label=legend_labels['optimized'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_subsample_gammas[i], c=color_map['subsampling'],
                                    label=legend_labels['subsampling'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_indp_gamma[i], c=color_map['data_indp'],
                                    label=legend_labels['data_indp'])
        axes[row_idx, col_idx].set_title('M = {}'.format(all_Ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].set_xticks([0, K / 2, K], [0, K / 2, K], fontsize=fontsize)
        axes[row_idx, col_idx].set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=fontsize)
        if not get_legend:
            axes[row_idx, col_idx].grid()
        if get_legend and i == 3:
            legend_fontsize = 15
            legend = axes[row_idx, col_idx].legend(fontsize=legend_fontsize, ncol=3, loc='lower right')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.text(0.5, 0.01, 'Support $l \in \{0,1,...,K\}$', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, '$\gamma$ values', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('Shape of $\gamma$ functions', fontsize=fontsize)

    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    # plt.show()
    plt.savefig('gamma_fn_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    no_exp = 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ## plot estimated TV distance
    for i in range(len(all_Ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=fontsize)
        axes[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=fontsize)
        # compute estimated TV distance
        list_gamma_functions = [all_opt_gammas[i], all_subsample_gammas[i], all_indp_gamma[i]]
        color_map_2 = [color_map['optimized'], color_map['subsampling'], color_map['data_indp']]
        all_results = np.zeros((len(list_gamma_functions), no_exp))
        for exp_idx in range(no_exp):
            for fn_idx in range(len(list_gamma_functions)):
                gamma_fn = list_gamma_functions[fn_idx]
                est_TV_distance = compute_estimated_TV_distance(gamma_fn)
                all_results[fn_idx, exp_idx] = est_TV_distance
        for fn_idx in range(len(list_gamma_functions)):
            mean_est_TV_dist = np.mean(all_results[fn_idx])
            std_est_TV_dist = np.std(all_results[fn_idx])
            # plot avg error of a specific funtion
            axes[row_idx, col_idx].scatter(fn_idx, mean_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_title('M = {}'.format(all_Ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].grid()
    # fig.text(0.5, 0.01, 'Error', ha='center', fontsize=fontsize)
    # fig.text(0.04, 0.5, '$\gamma$ functions', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.01, '$\gamma$ functions', ha='center', fontsize=fontsize)
    fig.text(0.01, 0.5, 'Error', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('$\mathcal{E}$(DaRRM$_{\gamma})$', fontsize=fontsize)
    # plt.show()
    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    plt.savefig('error_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()




def main_advanced_depracated():
    opt_gamma_folder = 'opt_gamma_fn'
    if not os.path.isdir(opt_gamma_folder):
        os.mkdir(opt_gamma_folder)
    K = 51
    eps = 0.1
    delta_prime = 0.01
    delta_0 = 0.01
    k = 15
    final_eps, delta, m = \
        compute_composed_privacy_and_failure_prob_by_subsamples(eps=eps, k=k, delta_0=delta_0, delta_prime=delta_prime)
    print('K : {}, delta_0: {}, k: {}'.format(K, delta_0, k))
    print('final eps: {}, delta: {}, m: {}'.format(final_eps, delta, m))
    opt_gamma_path = os.path.join(opt_gamma_folder,
                                  'opt_gamma_K_{}_delta_0_{}_eps_{}_k_{}.pkl'.format(K, delta_0, eps, k)
                                  .replace('.', '_'))
    if not os.path.isfile(opt_gamma_path):
        opt_gamma = optimize_gamma(K, eps, m, delta, delta_0)

        with open(opt_gamma_path, 'wb') as f:
            pickle.dump(opt_gamma, f)
    else:
        with open(opt_gamma_path, 'rb') as f:
            opt_gamma = pickle.load(f)




    # subsample_gamma = baseline_gamma(K, k)
    #
    # plot_gamma([opt_gamma, subsample_gamma], ['optimized', 'advanced subsampling'])
    #
    # # plot estimated TV distance
    # plot_estimated_TV_distance([opt_gamma, subsample_gamma], ['optimized', 'advanced subsampling'])


def compute_and_save_gamma():
    def compute_m(k, delta_prime, eps):
        return np.sqrt(2 * k * np.log(1 / delta_prime)) + k * (np.exp(eps) - 1)

    K = 35
    eps = 0.1
    all_Ms = [10, 13, 15, 20]
    delta_0 = 1e-5
    delta_prime = 0.1
    all_ms = [compute_m(k=M, delta_prime=delta_prime, eps=eps) for M in all_Ms]
    print('m: ', all_ms)
    for M in all_Ms:
        m = compute_m(k=M, delta_prime=delta_prime, eps=eps)
        delta = M * delta_0 + delta_prime
        print('m = {}, m eps: {}, delta: {}'.format(m, m * eps, delta))
        # optimized_gamma = optimize_gamma_gurobi(K, eps, m, delta, delta_0)
        # with open('opt_gamma_K_{}_eps_{}_m_{}_pure_DP.pkl'.format(K, eps, m), 'wb') as f:
        #     pickle.dump(optimized_gamma, f)


def main_non_uniform():
    K = 11
    eps = 0.1
    delta_0 = 0
    delta = 0

    m = 5

    opt_gamma_uniform = optimize_gamma_gurobi(K, eps, m, delta, delta_0, p_lower=0.5)
    opt_gamma_prior = optimize_gamma_gurobi(K, eps, m, delta, delta_0, p_lower=0.7)

    subsample_gamma = baseline_gamma(K, m)

    data_indp_gamma = RR_deta_independent_gamma(K=K, tau_eps=K * eps, m=m, eps=eps, lbda=K * delta_0, delta=delta)

    support = np.arange(0, K + 1)
    color_map = {
        'optimized_u': 'r',
        'optimized_p': 'm',
        'subsampling': 'b',
        'data_indp': 'g'
    }

    legend_labels_non_uniform = {
        'optimized_u': "$\gamma_{opt-U}$ (Ours)",
        'optimized_p': "$\gamma_{opt-P}$ (Ours)",
        'subsampling': "$\gamma_{Sub}$ (Baseline)",
        'data_indp': "$\gamma_{const}$ (Baseline)",
    }

    fontsize = 20
    ## plot gamma
    # plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # shape of gamma fn
    axes[0].tick_params(axis='both', which='major', labelsize=fontsize)
    axes[0].tick_params(axis='both', which='minor', labelsize=fontsize)
    axes[0].plot(support, opt_gamma_uniform, c=color_map['optimized_u'], label=legend_labels_non_uniform['optimized_u'])
    axes[0].plot(support, opt_gamma_prior, c=color_map['optimized_p'], label=legend_labels_non_uniform['optimized_p'])
    axes[0].plot(support, subsample_gamma, label=legend_labels_non_uniform['subsampling'], c=color_map['subsampling'])
    axes[0].plot(support, data_indp_gamma, label=legend_labels_non_uniform['data_indp'], c=color_map['data_indp'])
    axes[0].grid()
    # plt.legend(fontsize=15)
    # plt.xlabel('$\gamma$ support', fontsize=fontsize)
    axes[0].set_ylabel('$\gamma$ values', fontsize=fontsize)
    axes[0].xaxis.set_tick_params(labelsize=fontsize)
    axes[0].yaxis.set_tick_params(labelsize=fontsize)
    # axes[0, 0].set_xticklabels(fontsize=fontsize)
    # axes[0, 0].set_yticklabels(fontsize=fontsize)
    axes[0].set_xlabel('Support $l \in \{0,1,...,K\}$', fontsize=fontsize)
    axes[0].set_title('Shape of $\gamma$ functions', fontsize=fontsize)

    ## now plot E(darrm_gamma) under different actual p_i distributions
    p_dist_names = ["uniform", "disagree", "clear"]
    title_str = ["Actual: $p_i \sim$ Uniform([0, 1])", "Actual: $p_i = 0.5$", "Actual: $p_i \sim$ Uniform([0, 0.1])"]
    # if p_dist == "disagree":
    #     title_str = "Actual: $p_i = 0.5$"
    # elif p_dist == "clear":
    #     title_str = "Actual: $p_i \sim$ Uniform([0, 0.1])"
    # else:
    #     title_str = "Actual: $p_i \sim$ Uniform([0, 1])"
    # plt.title(title_str, fontsize=fontsize)
    for idx, p_dist in enumerate(p_dist_names):
        opt_gamma_uniform_tv_dist = compute_estimated_TV_distance(opt_gamma_uniform, p_dist=p_dist)
        opt_gamma_prior_tv_dist = compute_estimated_TV_distance(opt_gamma_prior, p_dist=p_dist)
        subsampling_gamma_tv_dist = compute_estimated_TV_distance(subsample_gamma, p_dist=p_dist)
        data_indp_gamma_tv_dist = compute_estimated_TV_distance(data_indp_gamma, p_dist=p_dist)
        axes[idx + 1].scatter(1, opt_gamma_uniform_tv_dist, c=color_map['optimized_u'],
                              label=legend_labels_non_uniform['optimized_u'])
        axes[idx + 1].scatter(2, opt_gamma_prior_tv_dist, c=color_map['optimized_p'],
                              label=legend_labels_non_uniform['optimized_p'])
        axes[idx + 1].scatter(3, subsampling_gamma_tv_dist, c=color_map['subsampling'],
                              label=legend_labels_non_uniform['subsampling'])
        axes[idx + 1].scatter(4, data_indp_gamma_tv_dist, c=color_map['data_indp'],
                              label=legend_labels_non_uniform['data_indp'])
        axes[idx + 1].grid()
        axes[idx + 1].set_xticks([])
        axes[idx + 1].yaxis.set_tick_params(labelsize=fontsize)
        axes[idx + 1].set_xlabel('$\gamma$ functions', fontsize=fontsize)
        axes[idx + 1].set_ylabel('Error', fontsize=fontsize)
        axes[idx + 1].set_title(title_str[idx], fontsize=fontsize)

    fig.suptitle("K={}, m={}, $\epsilon$=0.1".format(m, K), fontsize=fontsize)

    # plt.show()
    # plt.savefig('gamma_prior_m_{}_K_{}.pdf'.format(m, K), bbox_inches="tight", pad_inches=0.1)
    # plt.savefig('test', bbox_inches='tight', pad_inches=0.1)
    # plt.close()

    ## evaluate the utility of darrm with gamma
    # p_dist = "uniform"
    # # plt.figure(figsize=(10, 8))

    # plt.scatter(1, opt_gamma_uniform_tv_dist, label="Optimized DaRRM_$\gamma$ w/ $\mathcal{U}$ prior")
    # plt.scatter(2, opt_gamma_prior_tv_dist, label="Optimized DaRRM_$\gamma$ w/ $\mathcal{T} prior$")
    # plt.scatter(3, subsampling_gamma_tv_dist, label="Subsampling")
    # plt.scatter(4, data_indp_gamma_tv_dist, label="Randomized Response")
    # plt.grid()
    # ticksize = 20
    # # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.legend(fontsize=15)
    # plt.xlabel('', fontsize=fontsize)
    # plt.xticks([], [], fontsize=ticksize)
    # plt.yticks(fontsize=ticksize)
    # plt.ylabel('TV Distance', fontsize=fontsize)
    # if p_dist == "disagree":
    #     title_str = "Actual: $p_i = 0.5$"
    # elif p_dist == "clear":
    #     title_str = "Actual: $p_i \sim$ Uniform([0, 0.1])"
    # else:
    #     title_str = "Actual: $p_i \sim$ Uniform([0, 1])"
    # plt.title(title_str, fontsize=fontsize)
    # plt.show()
    # plt.savefig('TV_dist_pdist_{}_m_{}_K_{}.pdf'.format(p_dist, m, K), bbox_inches="tight", pad_inches=0.1)
    # plt.close()
    fig.tight_layout()
    plt.savefig('non_uniform_m_{}_K_{}.pdf'.format(m, K), bbox_inches="tight", pad_inches=0.1)
    plt.close()



if __name__ == '__main__':
    main_simple()
    # main_advanced()
    # compute_and_save_gamma()
    # main_non_uniform()
