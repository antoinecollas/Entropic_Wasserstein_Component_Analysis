from joblib import Parallel, delayed
import matplotlib
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter
import numpy as np
import numpy.linalg as la
import ot
import ot.plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sys import argv
from tqdm import tqdm

from data_loaders import check_dataset
from otpca import ot_pca_bcd
from utils import create_directory, save_figure


def wrapped_ot_pca_bcd(i, *args, **kwargs):
    G, P = ot_pca_bcd(*args, **kwargs)
    print(f'OT PCA {i} done')
    return G, P


def evaluate(
    X, y,
    test_size,
    n_splits_eval,
    n_neighbors,
    random_state=123
):
    misclassif = list()
    for i in range(n_splits_eval):
        X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=random_state + i,
                        shuffle=True,
                        stratify=y
                    )
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        misclassif.append(np.mean(y_pred != y_test))
    misclassif = np.array(misclassif)*100
    return round(np.mean(misclassif), 2), round(int(np.std(misclassif)), 2)


def main(
    X_full,
    y_full,
    dataset_name,
    project_in_span_X,
    random_state,
    test_size,
    n_splits_eval,
    k_list,
    reg_lists,
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
    plot_bary_proj,
    plot_transport_plan,
    plot_2d,
    perplexity,
    threshold_alpha,
    verbose
):
    folder_path = create_directory(f'genes_acc_vs_reg_{dataset_name}')

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

    # raw data
    vanilla_mean, vanilla_std = evaluate(
        X_full, y_full,
        test_size=test_size,
        n_splits_eval=n_splits_eval,
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    for k, reg_list in zip(k_list, reg_lists):
        print('=================================')
        print('k=', k)

        # PCA
        pca = PCA(n_components=k)
        pca.fit(X_full)
        X_full_pca = pca.transform(X_full)
        assert X_full_pca.shape == (n, k)
        pca_mean, pca_std = evaluate(
            X_full_pca, y_full,
            test_size=test_size,
            n_splits_eval=n_splits_eval,
            n_neighbors=n_neighbors,
            random_state=random_state
        )

        # OT PCA
        print(f'Computing OT PCA ({len(reg_list)} subspaces):')
        G_P = Parallel(n_jobs=n_jobs)(
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
            ) for i, reg in enumerate(reg_list)
        )

        G_list, P_list = list(), list()
        for G, P in G_P:
            G_list.append(G)
            P_list.append(P)

        ot_pca_mean, ot_pca_std = list(), list()
        ot_pca_bary_mean, ot_pca_bary_std = list(), list()
        for i, reg in enumerate(tqdm(reg_list)):
            X_full_ot_pca = X_full @ P_list[i]
            assert X_full_ot_pca.shape == (n, k)
            tmp_mean, tmp_std = evaluate(
                X_full_ot_pca, y_full,
                test_size=test_size,
                n_splits_eval=n_splits_eval,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            ot_pca_mean.append(tmp_mean)
            ot_pca_std.append(tmp_std)

            X_full_ot_pca = n * G_list[i] @ X_full @ P_list[i]
            assert X_full_pca.shape == (n, k)
            tmp_mean, tmp_std = evaluate(
                X_full_ot_pca, y_full,
                test_size=test_size,
                n_splits_eval=n_splits_eval,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            ot_pca_bary_mean.append(tmp_mean)
            ot_pca_bary_std.append(tmp_std)

        # plot misclassifications versus regularizations
        plt.figure()
        cmap = plt.get_cmap('tab10')
        color = cmap(0)
        plt.axhline(y=vanilla_mean, linestyle='--',
                    color=color, label='Raw data')
        color = cmap(1)
        plt.axhline(y=pca_mean, linestyle='--',
                    color=color, label='PCA')
        color = cmap(2)
        plt.semilogx(reg_list, ot_pca_mean, linestyle='-',
                     marker='+', color=color, label='OT PCA')
        if plot_bary_proj:
            color = cmap(3)
            plt.semilogx(reg_list, ot_pca_bary_mean, linestyle='-',
                         marker='x', color=color, label='OT PCA bary proj')
        plt.legend()
        # set major ticks format to integer
        ax = plt.gca()
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
        plt.xlabel(r'$\varepsilon$')
        plt.ylabel('Misclassification (%)')
        ylim_min = np.min([vanilla_mean, pca_mean, np.min(ot_pca_mean)]) - 1.5
        ylim_max = np.max([vanilla_mean, pca_mean, np.max(ot_pca_mean)]) + 3.5
        plt.ylim(ylim_min, ylim_max)
        plt.grid()

        save_figure(folder_path, f'misclassif_vs_reg_k_{str(k)}')

        # plot best optimal transport
        # BE CAREFULL: the following code assumes that data
        # from same classes are blockwise
        # ex: [0, 0, 0, 1, 1] and not [1, 0, 1, 1]
        if plot_transport_plan:
            # choose greatest reg value among those performing the best
            idx = np.max(np.argwhere(np.min(ot_pca_mean) == ot_pca_mean))
            best_reg = reg_list[idx]
            Gbcd, Pbcd = ot_pca_bcd(
                X_full, k, reg=best_reg,
                thresh=threshold_ot_pca,
                max_iter=max_iter_ot_pca,
                max_iter_sink=max_iter_sink,
                max_iter_MM=max_iter_MM,
                warn=warn, log=False,
                method_sink=method_sink,
                method=method,
                svd_fct_cpu=svd_fct_cpu,
                device=device, verbose=verbose
            )

            plt.figure()
            ax = plt.gca()
            cmap = plt.get_cmap('Blues')
            norm = matplotlib.colors.PowerNorm(.5, vmin=0, vmax=100)
            plt.imshow(n*Gbcd*100, cmap=cmap, norm=norm)
            for i, class_ in enumerate(np.sort(np.unique(y_full))):
                indices = (y_full == class_)
                idx_min = np.min(np.arange(len(y_full))[indices])
                idx_max = np.max(np.arange(len(y_full))[indices])
                width = idx_max - idx_min + 1
                rect = patches.Rectangle(
                    (idx_min-0.5, idx_min-0.5),
                    width, width,
                    linewidth=1, edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
            cb = plt.colorbar(fraction=0.046, pad=0.04)
            ticks_loc = cb.ax.get_yticks().tolist()
            cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            cb.ax.set_yticklabels([f'{int(i)}%' for i in cb.get_ticks()])
            plt.ylabel(r'($\mathbf{x}_1, \cdots, \mathbf{x}_n$)')
            x_label = r'($\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_1, \cdots,'
            x_label += r'\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_n$)'
            plt.xlabel(x_label)

            save_figure(
                folder_path, f'best_transport_plan_k_{k}_reg_{best_reg}')

            if PLOT_2D:
                X_proj = X_full@Pbcd
                viz = TSNE(
                    n_components=2, perplexity=perplexity,
                    learning_rate='auto', n_jobs=n_jobs,
                    random_state=random_state
                )
                X_2d = viz.fit_transform(X_proj)
                fig, ax = plt.subplots()
                Gbcd[Gbcd <= threshold_alpha] = 0
                ot.plot.plot2D_samples_mat(X_2d, X_2d, Gbcd)
                cmap = plt.get_cmap('tab10')
                for i, class_ in enumerate(np.unique(y_full)):
                    idx = (y_full == class_)
                    ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                               color=cmap(i), label=f'Class {str(class_)}')
                plt.legend()

                save_figure(folder_path, f'plot_2d_k_{k}_reg_{best_reg}')


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
    MAX_ITER_SINK = 100
    MAX_ITER_MM = 10
    MAX_ITER_OT_PCA = 10000
    N_NEIGHBORS = 1
    DEVICE = 'cpu'
    N_SPLITS_EVAL = 200
    TEST_SIZE = 0.5
    WARN = False

    PLOT_BARY_PROJ = False
    PLOT_TRANSPORT_PLAN = True
    PLOT_2D = True
    PERPLEXITY = 10

    if DATASET == 'khan2001':
        k_LIST = [5, 10]
        REG_LISTS = [
            np.geomspace(start=15, stop=80, num=20),
            np.geomspace(start=5, stop=170, num=20)
        ]
        THRESHOLD_ALPHA = 5*1e-4

    elif DATASET == 'Breast':
        k_LIST = [5, 25]
        REG_LISTS = [
            np.geomspace(start=50, stop=5000, num=20),
            np.geomspace(start=300, stop=4000, num=20)
        ]
        THRESHOLD_ALPHA = 4*1e-4

    X_full, y_full = load_data_fct(verbose=True)

    main(
        X_full=X_full,
        y_full=y_full,
        dataset_name=DATASET,
        project_in_span_X=PROJECT_IN_SPAN_X,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        n_splits_eval=N_SPLITS_EVAL,
        k_list=k_LIST,
        reg_lists=REG_LISTS,
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
        plot_bary_proj=PLOT_BARY_PROJ,
        plot_transport_plan=PLOT_TRANSPORT_PLAN,
        plot_2d=PLOT_2D,
        perplexity=PERPLEXITY,
        threshold_alpha=THRESHOLD_ALPHA,
        verbose=VERBOSE
    )
