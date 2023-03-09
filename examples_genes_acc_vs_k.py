import itertools
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sys import argv
from tqdm import tqdm

from data_loaders import check_dataset
from otpca import ot_pca_bcd
from utils import create_directory, save_figure


def wrapped_ot_pca_bcd(i, *args, **kwargs):
    _, P = ot_pca_bcd(*args, **kwargs)
    print(f'OT PCA {i} done')
    return P


def evaluate_projected_data(
    X_train, X_test, y_train, y_test,
    n_neighbors
):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    misclassif = np.mean(y_pred != y_test)
    return misclassif


def ot_pca_choose_best_reg(
    ot_pca_fitted,
    X_train,
    y_train,
    val_size,
    reg_list,
    n_splits_choose_reg,
    n_neighbors,
    random_state=123
):
    assert len(ot_pca_fitted) == len(reg_list)
    misclassif = np.zeros(shape=(n_splits_choose_reg, len(reg_list)))
    for j in range(len(ot_pca_fitted)):
        X_train_proj = X_train@ot_pca_fitted[j]
        for i in range(n_splits_choose_reg):
            # split train in (train/val)
            X_t, X_v, y_t, y_v = train_test_split(
                X_train_proj, y_train,
                test_size=val_size,
                random_state=random_state + i,
                shuffle=True,
                stratify=y_train
            )
            misclassif[i, j] = evaluate_projected_data(
                X_t, X_v, y_t, y_v,
                n_neighbors=n_neighbors
            )
    misclassif = np.mean(misclassif, axis=0)
    # choose greatest reg value among those performing the best
    idx = np.max(np.argwhere(np.min(misclassif) == misclassif))
    P = ot_pca_fitted[idx]
    best_reg = reg_list[idx]
    return P, best_reg


def eval_one_split(
    i,
    X_full,
    y_full,
    pca_fitted,
    ot_pca_fitted,
    random_state,
    val_size,
    test_size,
    n_splits_choose_reg,
    k_list,
    reg_list,
    n_neighbors,
    device
):
    # split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=test_size,
        random_state=random_state + i,
        shuffle=True,
        stratify=y_full
    )

    # init arrays
    pca_mis = np.zeros(len(k_list))
    ot_pca_mis = np.zeros(len(k_list))
    ot_pca_best_reg = np.zeros(len(k_list))

    # raw data
    vanilla_mis = evaluate_projected_data(
        X_train, X_test,
        y_train, y_test,
        n_neighbors=n_neighbors
    )

    for j, k in enumerate(k_list):
        # PCA
        X_train_pca = pca_fitted[j].transform(X_train)
        X_test_pca = pca_fitted[j].transform(X_test)
        assert X_train_pca.shape == (X_train.shape[0], k)
        assert X_test_pca.shape == (X_test.shape[0], k)
        pca_mis[j] = evaluate_projected_data(
            X_train_pca, X_test_pca,
            y_train, y_test,
            n_neighbors=n_neighbors
        )

        # OT-PCA
        P, best_reg = ot_pca_choose_best_reg(
            ot_pca_fitted[j],
            X_train, y_train,
            val_size=val_size,
            reg_list=reg_list,
            n_splits_choose_reg=n_splits_choose_reg,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        ot_pca_best_reg[j] = best_reg
        X_train_ot_pca = X_train@P
        X_test_ot_pca = X_test@P
        assert X_train_ot_pca.shape == (X_train.shape[0], k)
        assert X_test_ot_pca.shape == (X_test.shape[0], k)
        ot_pca_mis[j] = evaluate_projected_data(
            X_train_ot_pca, X_test_ot_pca,
            y_train, y_test,
            n_neighbors=n_neighbors
        )

    print(f'Eval {i} done.')

    return vanilla_mis, pca_mis, ot_pca_mis, ot_pca_best_reg


def main(
    X_full,
    y_full,
    dataset_name,
    project_in_span_X,
    random_state,
    val_size,
    test_size,
    n_splits_choose_reg,
    n_splits_eval,
    k_list,
    reg_list,
    method,
    svd_fct_cpu,
    method_sink,
    max_iter_sink,
    max_iter_MM,
    max_iter_ot_pca,
    threshold_ot_pca,
    warn,
    n_neighbors,
    n_jobs,
    device,
    verbose
):
    folder_path = create_directory(f'genes_acc_vs_k_{dataset_name}')

    n = X_full.shape[0]

    # center data
    X_full = X_full - X_full.mean(0)

    # for faster computation on the dataset: project in span(X)
    if project_in_span_X:
        U, _, _ = la.svd(X_full.T, full_matrices=False)
        # n centered data span a vector space of dimension n-1
        U = U[:, :-1]
        # check reconstruction error
        assert la.norm(X_full - X_full @ U @ U.T) / (n**2) <= 1e-10
        X_full = X_full @ U
        assert X_full.shape == (n, n-1)

    # computing all subspaces
    print(f'Computing PCA ({len(k_list)} subspaces):')
    pca_fitted = list()
    for k in tqdm(k_list):
        pca = PCA(n_components=k)
        pca.fit(X_full)
        pca_fitted.append(pca)

    print(f'Computing OT PCA ({len(k_list)*len(reg_list)} subspaces):')
    res = Parallel(n_jobs=n_jobs)(
        delayed(wrapped_ot_pca_bcd)(
            i,
            X_full, k, reg=reg,
            thresh=threshold_ot_pca,
            max_iter=max_iter_ot_pca,
            max_iter_sink=max_iter_sink,
            max_iter_MM=max_iter_MM,
            warn=warn, log=False,
            method_sink=method_sink,
            method=method,
            svd_fct_cpu=svd_fct_cpu,
            device=device, verbose=verbose
        ) for i, (k, reg) in enumerate(itertools.product(k_list, reg_list))
    )

    ot_pca_fitted = list()
    for i, k in enumerate(k_list):
        ot_pca_fitted.append(list())
        for j, reg in enumerate(reg_list):
            ot_pca_fitted[i].append(res[i*len(reg_list)+j])

    # evaluation of all subspaces
    print(f'Evaluation: ({n_splits_eval} splits)')

    res = Parallel(n_jobs=n_jobs)(
        delayed(eval_one_split)(
            i=i,
            X_full=X_full,
            y_full=y_full,
            pca_fitted=pca_fitted,
            ot_pca_fitted=ot_pca_fitted,
            random_state=random_state,
            val_size=val_size,
            test_size=test_size,
            n_splits_choose_reg=n_splits_choose_reg,
            k_list=k_list,
            reg_list=reg_list,
            n_neighbors=n_neighbors,
            device=device
        ) for i in range(n_splits_eval)
    )
    vanilla_misclassif, pca_misclassif = list(), list()
    ot_pca_misclassif, ot_pca_best_reg = list(), list()
    for i in range(len(res)):
        vanilla_misclassif.append(res[i][0])
        pca_misclassif.append(res[i][1])
        ot_pca_misclassif.append(res[i][2])
        ot_pca_best_reg.append(res[i][3])

    vanilla_misclassif = np.array(vanilla_misclassif)*100
    pca_misclassif = np.stack(pca_misclassif, axis=0)*100
    ot_pca_misclassif = np.stack(ot_pca_misclassif, axis=0)*100
    ot_pca_best_reg = np.stack(ot_pca_best_reg, axis=0)

    # plot best regularization parameter versus k
    plt.figure(0)
    plt.axhline(y=np.max(reg_list), linestyle='--',
                color='blue', label='Maximum tested value')
    plt.axhline(y=np.min(reg_list), linestyle='--',
                color='green', label='Minimum tested value')
    plt.boxplot(ot_pca_best_reg, labels=k_list)
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'Best $\varepsilon$')
    ylim_max = 1.2*np.max(reg_list)
    plt.ylim(None, ylim_max)

    save_figure(folder_path, 'best_reg_vs_k')

    cmap = plt.get_cmap('tab10')

    # plot misclassifications versus k
    plt.figure(2)
    plt.axhline(
        y=np.mean(vanilla_misclassif), linestyle='--',
        color=cmap(0), label='Raw data')
    if dataset_name == 'Breast':
        plt_fct = plt.semilogx
    else:
        plt_fct = plt.plot
    plt_fct(
        k_list, np.mean(pca_misclassif, axis=0),
        marker='x', color=cmap(1), label='PCA')
    plt_fct(
        k_list, np.mean(ot_pca_misclassif, axis=0),
        marker='+', color=cmap(2), label='OT-PCA')
    plt.xticks(k_list, k_list)

    alpha = 0.25
    # add fill between 0.25 and 0.75 quantiles
    q1, q2 = 0.25, 0.75
    plt.fill_between(
        k_list,
        np.quantile(vanilla_misclassif, q1),
        np.quantile(vanilla_misclassif, q2),
        alpha=alpha, color=cmap(0))
    plt.fill_between(
        k_list,
        np.quantile(pca_misclassif, q1, axis=0),
        np.quantile(pca_misclassif, q2, axis=0),
        alpha=alpha, color=cmap(1))
    plt.fill_between(
        k_list,
        np.quantile(ot_pca_misclassif, q1, axis=0),
        np.quantile(ot_pca_misclassif, q2, axis=0),
        alpha=alpha, color=cmap(2))

    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel('Misclassification (%)')
    ylim_max = 1.2*np.max(
        [
            np.mean(vanilla_misclassif),
            np.max(np.mean(ot_pca_misclassif, axis=0))
        ]
    )
    plt.ylim(None, ylim_max)
    plt.grid()

    save_figure(folder_path, 'misclassification_vs_k')

    # plot misclassifications versus k using group boxplots
    # select only a few k values
    k_list = k_list[::2]
    pca_misclassif = pca_misclassif[:, ::2]
    ot_pca_misclassif = ot_pca_misclassif[:, ::2]
    # repeat k_list and vanilla_misclassif to have
    # the same number of elements as ot_pca_misclassif
    vanilla_misclassif = np.repeat(
        vanilla_misclassif[:, np.newaxis], len(k_list), axis=1)
    k_list = np.repeat(k_list[np.newaxis, :], n_splits_eval, axis=0)
    df = pd.DataFrame(
        {
            'k': k_list.flatten(),
            'Raw data': vanilla_misclassif.flatten(),
            'PCA': pca_misclassif.flatten(),
            'OT-PCA': ot_pca_misclassif.flatten()
        }
    )
    df = pd.melt(
        df, id_vars=['k'], var_name='Method',
        value_name='Misclassification (%)')
    # plot
    plt.figure(3)
    colors = [cmap(0), cmap(1), cmap(2)]
    sns.boxplot(
        x='k', y='Misclassification (%)', hue='Method',
        data=df, palette=colors,
        meanline=True, showmeans=True,
        medianprops={'visible': False},
        meanprops={'color': 'k', 'ls': '-', 'lw': 2}
    )
    plt.xlabel(r'$k$')
    plt.ylabel('Misclassification (%)')
    plt.ylim(None, 1.5*ylim_max)

    save_figure(folder_path, 'misclassification_vs_k_box_plot')


if __name__ == '__main__':
    N_JOBS = -1
    VERBOSE = False

    DATASET = argv[1]
    load_data_fct = check_dataset(DATASET)
    print(f'Dataset: {DATASET}')

    RANDOM_STATE = 123
    METHOD = 'MM'
    SVD_FCT_CPU = 'numpy'
    PROJECT_IN_SPAN_X = True
    METHOD_SINK = 'sinkhorn_stabilized'
    THRESHOLD_OT_PCA = 1e-4
    MAX_ITER_SINK = 20
    MAX_ITER_MM = 10
    MAX_ITER_OT_PCA = 10000
    WARN = False
    N_NEIGHBORS = 1
    DEVICE = 'cpu'
    N_SPLITS_CHOOSE_REG = 20
    N_SPLITS_EVAL = 100
    VAL_SIZE = 0.5
    TEST_SIZE = 0.5

    # BE CAREFUL:
    # k should be not too large otherwise
    # KNeighborsClassifier becomes very slow.

    if DATASET == 'khan2001':
        k_LIST = np.array(list(range(4, 13)))
        REG_LIST = np.geomspace(start=5, stop=200, num=20)

    elif DATASET == 'Breast':
        k_LIST = np.array(list(range(3, 11)) + list(range(15, 41, 5)))
        REG_LIST = np.geomspace(start=50, stop=5000, num=20)

    X_full, y_full = load_data_fct(verbose=True)

    main(
        X_full,
        y_full,
        dataset_name=DATASET,
        project_in_span_X=PROJECT_IN_SPAN_X,
        random_state=RANDOM_STATE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        n_splits_choose_reg=N_SPLITS_CHOOSE_REG,
        n_splits_eval=N_SPLITS_EVAL,
        k_list=k_LIST,
        reg_list=REG_LIST,
        method=METHOD,
        svd_fct_cpu=SVD_FCT_CPU,
        method_sink=METHOD_SINK,
        max_iter_sink=MAX_ITER_SINK,
        max_iter_MM=MAX_ITER_MM,
        max_iter_ot_pca=MAX_ITER_OT_PCA,
        threshold_ot_pca=THRESHOLD_OT_PCA,
        warn=WARN,
        n_neighbors=N_NEIGHBORS,
        n_jobs=N_JOBS,
        device=DEVICE,
        verbose=VERBOSE
    )
