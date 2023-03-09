import numpy as np
from numpy import random as rnd
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
import pylab as plt

from otpca import ot_pca_bcd
from utils import create_directory, save_figure


def plot_scatter_subspace(X, c_y, subspace, fig):
    # 3d plot
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=c_y)
    subspace_perp = np.cross(subspace[:, 0], subspace[:, 1])
    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])
    min_z, max_z = np.min(X[:, 2]), np.max(X[:, 2])
    interval_x = np.linspace(min_x-1, max_x+1, num=10)
    interval_y = np.linspace(min_y-1, max_y+1, num=10)
    x, y = np.meshgrid(interval_x, interval_y)
    z = (- subspace_perp[0]*x - subspace_perp[1]*y) / subspace_perp[2]
    ax.plot_surface(x, y, z, alpha=0.4, color='orange', linewidth=0)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    ax.set_zlim3d(min_z, max_z)

    # 2d plot
    ax = fig.add_subplot(122)
    X_proj = X@subspace
    ax.scatter(X_proj[:, 0], X_proj[:, 1], color=c_y)


def main(method, interactive):
    rnd.seed(123)

    folder_path = create_directory('3d_example')

    n = 50
    d = 3
    k = 2

    reg_list = [1, 100]
    max_iter_sink = 100

    # data generation
    Q = ortho_group.rvs(d)
    D = np.diag(np.abs(rnd.normal(size=d)))
    cov_sqrtm = Q@D@Q.T
    X = rnd.normal(size=(n, d))@cov_sqrtm
    y = np.zeros(n)
    X = np.concatenate([X, rnd.normal(size=(n, d))@cov_sqrtm + 1], axis=0)
    y = np.concatenate([y, np.ones(n)])
    X = X - np.mean(X, axis=0)

    # plot
    c_y = np.array(['blue']*2*n)
    c_y[y == 0] = 'red'

    # plot PCA
    pca = PCA(n_components=k)
    pca.fit(X)

    fig = plt.figure(0)
    plot_scatter_subspace(X, c_y, pca.components_.T, fig)
    title = 'PCA'
    plt.title(title)
    if not interactive:
        save_figure(folder_path, title)

    # plot OT PCA
    for i, reg in enumerate(reg_list):
        Gbcd, Pbcd, log_bcd = ot_pca_bcd(
            X, k=k, reg=reg, verbose=True,
            method=method, svd_fct_cpu='numpy',
            max_iter_sink=max_iter_sink)

        fig = plt.figure(i+1)
        plot_scatter_subspace(X, c_y, Pbcd, fig)
        title = f'OT PCA ({method}), reg={str(reg)}'
        plt.title(title)
        if not interactive:
            save_figure(folder_path, title)

    if interactive:
        plt.show()


if __name__ == '__main__':
    METHOD = 'MM'
    INTERACTIVE = False

    main(method=METHOD, interactive=INTERACTIVE)
