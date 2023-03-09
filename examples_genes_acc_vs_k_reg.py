import itertools
from joblib import Parallel, delayed
from matplotlib import pyplot as plt, ticker as mticker
import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sys import argv

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


def eval_one_split(
    number,
    X_full,
    y_full,
    ot_pca_fitted,
    random_state,
    test_size,
    k_list,
    reg_list,
    n_neighbors
):
    # split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=test_size,
        random_state=random_state + number,
        shuffle=True,
        stratify=y_full
    )

    # evaluate on all k
    ot_pca_mis = np.zeros((len(k_list), len(reg_list)))
    for i, k in enumerate(k_list):
        for j, reg in enumerate(reg_list):
            P = ot_pca_fitted[i][j]
            X_train_proj = X_train @ P
            X_test_proj = X_test @ P
            ot_pca_mis[i, j] = evaluate_projected_data(
                X_train_proj, X_test_proj, y_train, y_test,
                n_neighbors
            )

    print(f'Eval {number} done.')

    return ot_pca_mis


def main(
    X_full,
    y_full,
    dataset_name,
    project_in_span_X,
    random_state,
    test_size,
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
    folder_path = create_directory(f'genes_acc_vs_k_reg_{dataset_name}')

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
            number=i,
            X_full=X_full,
            y_full=y_full,
            ot_pca_fitted=ot_pca_fitted,
            random_state=random_state,
            test_size=test_size,
            k_list=k_list,
            reg_list=reg_list,
            n_neighbors=n_neighbors
        ) for i in range(n_splits_eval)
    )
    res = np.stack(res, axis=0)

    # plot 2D results (k, reg) vs misclassification
    ot_pca_mis = np.mean(res, axis=0) * 100
    fig, ax = plt.subplots()
    im = ax.imshow(ot_pca_mis, cmap='viridis')
    ax.set_xticks(np.arange(len(reg_list)))
    ax.set_yticks(np.arange(len(k_list)))
    ax.set_xticklabels(reg_list.astype(int), rotation=90)
    ax.set_yticklabels(k_list)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$k$')
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks_loc = cb.ax.get_yticks().tolist()
    cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cb.ax.set_yticklabels([f'{int(i)}%' for i in cb.get_ticks()])
    save_figure(folder_path, 'misclassification_vs_k_reg')
    plt.close()


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
    N_SPLITS_EVAL = 100
    TEST_SIZE = 0.5

    # BE CAREFUL:
    # k should be not too large otherwise
    # KNeighborsClassifier becomes very slow.

    if DATASET == 'khan2001':
        k_LIST = np.array(list(range(4, 13)))
        REG_LIST = np.geomspace(start=5, stop=200, num=20)

    elif DATASET == 'Breast':
        k_LIST = np.array(list(range(3, 11)) + list(range(15, 41, 5)))
        REG_LIST = np.geomspace(start=350, stop=3500, num=20)

    X_full, y_full = load_data_fct(verbose=True)

    main(
        X_full,
        y_full,
        dataset_name=DATASET,
        project_in_span_X=PROJECT_IN_SPAN_X,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
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
