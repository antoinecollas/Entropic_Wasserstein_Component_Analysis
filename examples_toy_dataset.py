import numpy as np
from numpy import random as rnd
from sklearn.decomposition import PCA
from otpca import ot_pca_bcd, ot_pca_auto_diff
import pylab as plt

from utils import create_directory, save_figure


def main():
    folder_path = create_directory('toy_dataset')

    rnd.seed(123)

    n = 100
    d = 20

    X = 0.1*rnd.randn(n, d)
    X[:33, 0] += 1
    X[33:66, 1] += -1
    X = X-X.mean(0)

    reg = 0.1
    max_iter_sink = 100
    max_iter_MM = 20
    lr = 1e-1

    Gbcd, Pbcd, log_bcd = ot_pca_bcd(
        X, k=2, method='MM',
        reg=reg, verbose=False,
        max_iter_sink=max_iter_sink,
        max_iter_MM=max_iter_MM
    )
    Gauto, Pauto, log_auto = ot_pca_auto_diff(
        X, k=2, reg=reg, lr=lr, max_iter=100,
        max_iter_sink=max_iter_sink,
        verbose=False, log=True,
        device='cpu'
    )

    pca = PCA(n_components=2)
    pca.fit(X)

    xpca = pca.transform(X)

    xspca_bcd = X.dot(Pbcd)
    xspca2_bcd = n*Gbcd.dot(X.dot(Pbcd))

    xspca_auto = X.dot(Pauto)
    xspca2_auto = n*Gauto.dot(X.dot(Pauto))

    # plot PCA, OT PCA, shrinked OT PCA
    plt.figure(1, (15, 7))
    plt.subplot(2, 3, 1)
    plt.scatter(xpca[:, 0], xpca[:, 1])
    plt.title('PCA')

    plt.subplot(2, 3, 2)
    plt.scatter(xspca_bcd[:, 0], xspca_bcd[:, 1])
    plt.title('Sinkhorn PCA (BCD)')

    plt.subplot(2, 3, 3)
    plt.scatter(xspca2_bcd[:, 0], xspca2_bcd[:, 1])
    plt.title('Sinkhorn PCA shrinked (BCD)')

    plt.subplot(2, 3, 5)
    plt.scatter(xspca_auto[:, 0], xspca_auto[:, 1])
    plt.title('OT PCA (Auto-diff)')

    plt.subplot(2, 3, 6)
    plt.scatter(xspca2_auto[:, 0], xspca2_auto[:, 1])
    plt.title('OT PCA shrinked (Auto-diff)')

    save_figure(folder_path, 'plot_data_2d')

    # plot loss
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(Gbcd)
    plt.title('G (BCD)')
    plt.subplot(1, 2, 2)
    plt.imshow(Gauto)
    plt.title('G (Auto-diff)')

    plt.figure(3, (15, 7))
    ax = plt.subplot(1, 3, 1)
    ax.ticklabel_format(useOffset=False)
    plt.plot(np.arange(1, len(log_bcd['loss'])+1), log_bcd['loss'])
    plt.title('Loss BCD')
    ax = plt.subplot(1, 3, 2)
    ax.ticklabel_format(useOffset=False)
    plt.plot(np.arange(1, len(log_auto['loss'])+1), log_auto['loss'])
    plt.title('Loss Riemannian')
    ax = plt.subplot(1, 3, 3)
    ax.ticklabel_format(useOffset=False)
    plt.loglog(np.arange(1, len(log_auto['gradnorm'])+1), log_auto['gradnorm'])
    plt.title('Riemannian gradnorm')

    save_figure(folder_path, 'plot_loss')


if __name__ == '__main__':
    main()
