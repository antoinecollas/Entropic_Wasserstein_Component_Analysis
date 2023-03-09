from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from data_loaders import load_Breast
from otpca import ot_pca_auto_diff, ot_pca_bcd
from utils import create_directory, save_figure


def one_sample(
    random_state,
    compute_autodiff,
    X_full,
    y_full,
    dataset_name,
    k,
    p_list,
    reg_list,
    bcd_p_max,
    lr,
    method_sink,
    svd_fct_cpu,
    max_iter_sink,
    max_iter_MM,
    max_iter_otpca,
    thresh_crit,
    warn,
    device,
    verbose
):
    np.random.seed(random_state)

    n, p_full = X_full.shape

    # center data
    X_full = X_full - X_full.mean(0)

    # measure computational times
    tbcd_list, tmm_list, tauto_list = list(), list(), list()

    for p, reg in zip(tqdm(p_LIST), reg_list):
        if verbose:
            print(f'p={p}')

        # sample genes
        if p < p_full:
            indices = np.arange(p_full)
            np.random.shuffle(indices)
            indices = indices[:p]
            indices = np.sort(indices)
            X_reduced = X_full[:, indices]
        else:
            X_reduced = X_full

        # otpca autodiff
        if compute_autodiff:
            t0 = time.time()
            _, _ = ot_pca_auto_diff(
                X_reduced, k, reg=reg, lr=lr,
                thresh=thresh_crit,
                max_iter=max_iter_otpca,
                max_iter_sink=max_iter_sink,
                log=False, method_sink=method_sink,
                device=device, warn=warn,
                verbose=verbose
            )
            t1 = time.time()
            tauto = t1 - t0
            tauto_list.append(tauto)
        else:
            tauto = None

        # otpca BCD
        if p <= bcd_p_max:
            t0 = time.time()
            _, _ = ot_pca_bcd(
                X_reduced, k, reg=reg, thresh=thresh_crit,
                max_iter=max_iter_otpca, max_iter_sink=max_iter_sink,
                verbose=verbose, log=False, method_sink=method_sink,
                method='BCD', svd_fct_cpu=svd_fct_cpu, device='cpu',
                warn=warn
            )
            t1 = time.time()
            tbcd = t1 - t0
            tbcd_list.append(tbcd)
        else:
            tbcd = None

        # otpca MM
        t0 = time.time()
        _, _ = ot_pca_bcd(
            X_reduced, k, reg=reg, thresh=thresh_crit,
            max_iter=max_iter_otpca, max_iter_sink=max_iter_sink,
            max_iter_MM=max_iter_MM, verbose=verbose, log=False,
            method_sink=method_sink, method='MM', svd_fct_cpu=svd_fct_cpu,
            device='cpu', warn=warn
        )
        t1 = time.time()
        tmm = t1 - t0
        tmm_list.append(tmm)

        # print computation times
        if verbose:
            if tauto is not None:
                print(f'Elapsed time Riemannian: {tauto:,.1f}s')
            if tbcd is not None:
                print(f'Elapsed time BCD: {tbcd:,.1f}s')
            print(f'Elapsed time MM: {tmm:,.1f}s')
            print()

    return np.array(tauto_list), np.array(tbcd_list), np.array(tmm_list)


def main(
    random_state,
    n_MC,
    n_jobs,
    plot,
    plot_autodiff,
    **kwargs
):
    folder_path = create_directory('genes_compute_time_vs_p')

    res = Parallel(n_jobs=n_jobs)(
        delayed(one_sample)(
            random_state=random_state+i,
            compute_autodiff=plot_autodiff,
            **kwargs
        ) for i in range(n_MC)
    )
    tauto = np.zeros((n_MC, len(res[0][0])))
    tbcd = np.zeros((n_MC, len(res[0][1])))
    tmm = np.zeros((n_MC, len(res[0][2])))
    for i in range(n_MC):
        tauto[i, :] = res[i][0]
        tbcd[i, :] = res[i][1]
        tmm[i, :] = res[i][2]

    if plot:
        p_list = np.array(kwargs['p_list'])

        plt.figure()
        cmap = plt.get_cmap('tab10')

        # autodiff
        if plot_autodiff:
            color = cmap(0)
            mean = np.mean(tauto, axis=0)
            q025 = np.quantile(tauto, 0.25, axis=0)
            q075 = np.quantile(tauto, 0.75, axis=0)
            plt.loglog(p_list, mean, color=color, label='Riemannian')
            plt.fill_between(p_list, q025, q075, color=color, alpha=.25)

        # BCD
        color = cmap(1)
        mean = np.mean(tbcd, axis=0)
        q025 = np.quantile(tbcd, 0.25, axis=0)
        q075 = np.quantile(tbcd, 0.75, axis=0)
        plt.loglog(p_list[:len(mean)], mean, color=color, label='BCD')
        plt.fill_between(
            p_list[:len(mean)], q025, q075, color=color, alpha=.25)

        # MM
        color = cmap(2)
        mean = np.mean(tmm, axis=0)
        q025 = np.quantile(tmm, 0.25, axis=0)
        q075 = np.quantile(tmm, 0.75, axis=0)
        plt.loglog(p_list, mean, color=color, label='MM')
        plt.fill_between(p_list, q025, q075, color=color, alpha=.25)

        plt.xlabel(r'$d$')
        plt.ylabel('Computation time (s.)')
        plt.xlim(40, 60000)
        # plt.ylim(0.025, 100)
        plt.grid()
        plt.legend()

        save_figure(folder_path, 'computation_time')


if __name__ == '__main__':
    # Perform MAX_ITER_OTPCA iterations for each method
    # and compare computation times.
    # Verbose=True and N_MC=1 help to monitor that
    # both methods (BCD and MM) achieve same loss values
    # for all k.

    N_JOBS = -1
    RANDOM_STATE = 123
    VERBOSE = False

    DATASET = 'Breast'

    if DATASET == 'Breast':
        N_MC = 100
        k = 10
        p_LIST = np.geomspace(start=50, stop=54675, num=12).astype(int)
        REG_LIST = p_LIST / 100
        BCD_p_MAX = 2500
        METHOD_SINK = 'sinkhorn_stabilized'
        SVD_FCT_CPU = 'numpy'
        MAX_ITER_SINK = 100
        MAX_ITER_MM = 20
        MAX_ITER_OTPCA = 50
        THRESH_CRIT = 0
        WARN = False
        DEVICE = 'cpu'
        LR = 0.5
        PLOT = True
        PLOT_AUTODIFF = False

        X_full, y_full = load_Breast(verbose=True)

        print(f'Dataset: {DATASET}')
        print(f'List of p values: {p_LIST}')

        main(
            random_state=RANDOM_STATE,
            n_MC=N_MC,
            n_jobs=N_JOBS,
            plot=PLOT,
            plot_autodiff=PLOT_AUTODIFF,
            X_full=X_full,
            y_full=y_full,
            dataset_name=DATASET,
            k=k,
            p_list=p_LIST,
            reg_list=REG_LIST,
            bcd_p_max=BCD_p_MAX,
            lr=LR,
            method_sink=METHOD_SINK,
            svd_fct_cpu=SVD_FCT_CPU,
            max_iter_sink=MAX_ITER_SINK,
            max_iter_MM=MAX_ITER_MM,
            max_iter_otpca=MAX_ITER_OTPCA,
            thresh_crit=THRESH_CRIT,
            warn=WARN,
            device=DEVICE,
            verbose=VERBOSE
        )
