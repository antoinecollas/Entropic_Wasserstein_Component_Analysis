import numpy as np
from numpy import random as rnd
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
import pylab as plt

from otpca import ot_pca_bcd
from utils import create_directory, save_figure


def plot_scatter_subspace(X, c_y, subspace):
    axes = plt.gca()
    axes.set_xlim([-5, 5])
    axes.set_ylim([-5, 5])
    plt.scatter(X[:, 0], X[:, 1], color=c_y)
    axes.set_aspect('equal', adjustable='box')
    x_vals = np.array(axes.get_xlim())
    slope = subspace[1]/subspace[0]
    y_vals = slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def main(method):
    rnd.seed(123)

    folder_path = create_directory('2d_example')

    n = 200
    d = 2
    k = 1

    Q = ortho_group.rvs(d)
    D = np.diag(np.abs(rnd.normal(size=d)))
    cov = Q@D@Q.T
    X = rnd.normal(size=(n, d))@cov
    y = np.zeros(n)
    X = np.concatenate([X, rnd.normal(size=(n, d))@cov + 1], axis=0)
    y = np.concatenate([y, np.ones(n)])
    X = X - np.mean(X, axis=0)

    reg = 10
    max_iter_sink = 100

    Gbcd, Pbcd, log_bcd = ot_pca_bcd(
        X, k=k, reg=reg, verbose=True,
        svd_fct_cpu='numpy',
        method=method, max_iter_sink=max_iter_sink)

    pca = PCA(n_components=k)
    pca.fit(X)

    c_y = np.array(['blue']*2*n)
    c_y[y == 0] = 'red'

    plt.figure(1, (15, 7))
    plt.subplot(1, 2, 1)
    plot_scatter_subspace(X, c_y, pca.components_[0])
    plt.title('PCA')
    plt.subplot(1, 2, 2)
    plot_scatter_subspace(X, c_y, Pbcd)
    plt.title(f'OT PCA ({method})')
    save_figure(folder_path, 'subspaces')

    plt.figure(2)
    plt.imshow(Gbcd)
    plt.title(f'Transport plan ({method})')
    save_figure(folder_path, 'transport_plan')

    plt.figure(3)
    plt.plot(np.arange(1, len(log_bcd['loss'])+1), log_bcd['loss'])
    plt.title(f'Loss {method}')
    save_figure(folder_path, 'loss')


if __name__ == '__main__':
    METHOD = 'MM'
    main(method=METHOD)
