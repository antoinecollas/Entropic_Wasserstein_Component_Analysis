# %% Figure illustration of the method
import numpy as np
from numpy import random as rnd
from sklearn.datasets import make_blobs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.patches as patches
import matplotlib

from otpca import ot_pca_bcd
from utils import create_directory, save_figure

folder_path = create_directory('example_gmm')
cmap = matplotlib.colormaps.get_cmap('tab10')

rnd.seed(123)
n_samples = [15, 15]
esp = 0.8
centers = np.array([
                    [esp, esp],
                    [-esp, -esp]
                    ])
cluster_std = [0.4, 0.4]
X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    cluster_std=cluster_std,
    shuffle=False
)
X = X-X.mean(0)
n = X.shape[0]
reg = 0.3
G, P = ot_pca_bcd(
    X,
    2,
    reg=reg,
    log=False,
    method_sink='sinkhorn',
    method='BCD',
    max_iter_sink=1000
)
u = P[:, 1]
fs = 15
scale = 3
origin = np.array([0, 0])

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for k in [0]:
    u = P[:, k]
    ax[0].plot(
        [origin[0], scale * u[0] + origin[0]],
        [origin[1], scale*u[1] + origin[1]],
        color='grey',
        linestyle='--',
        lw=3,
        alpha=0.3
    )
    label_ = r'$\mathbf{U}$' if k == 0 else None
    ax[0].plot(
        [origin[0], -scale * u[0] + origin[0]],
        [origin[1], -scale*u[1] + origin[1]],
        color='grey',
        linestyle='--',
        lw=3,
        alpha=0.3,
        label=label_
    )
u = P[:, 0]
X1 = X @ u[:, None] @ u[:, None].T + origin

ax[0].axis('scaled')
thresh = 0.15
mm = 1
for i in range(n):
    for j in range(n):
        v = G[i, j] / G.max()
        if v >= thresh or (i, j) == (n-1, n-1):
            ax[0].plot(
                [X[i, 0], X1[j, 0]],
                [X[i, 1], X1[j, 1]],
                alpha=mm * v,
                linestyle='-',
                c='C0',
                label=r'$\pi_{ij}$' if (i, j) == (n - 1, n - 1) else None
            )
ax[0].scatter(
    X[:, 0], X[:, 1],
    color=[cmap(y[i] + 1) for i in range(n)],
    alpha=0.4,
    zorder=30,
    s=50
)
ax[0].scatter(
    X1[:, 0], X1[:, 1], color=[cmap(y[i] + 1) for i in range(n)],
    alpha=0.9,
    s=50,
    marker='+',
    label=r'$\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_i$',
    zorder=30
)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].legend(fontsize=fs, loc='upper left')

divider = make_axes_locatable(ax[1])
norm = matplotlib.colors.PowerNorm(.5, vmin=0, vmax=100)
im = ax[1].imshow(n * G * 100, cmap=plt.cm.Blues, norm=norm, aspect="auto")
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation='vertical')
ticks_loc = cb.ax.get_yticks().tolist()
cb.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
cb.ax.set_yticklabels([f'{int(i)}%' for i in cb.get_ticks()])
cb.ax.tick_params(labelsize=fs)
for i, class_ in enumerate(np.sort(np.unique(y))):
    indices = (y == class_)
    idx_min = np.min(np.arange(len(y))[indices])
    idx_max = np.max(np.arange(len(y))[indices])
    width = idx_max - idx_min + 1
    rect = patches.Rectangle(
        (idx_min - 0.5, idx_min - 0.5),
        width, width,
        linewidth=1, edgecolor='r',
        facecolor='none'
    )
    ax[1].add_patch(rect)

ax[1].set_title('OT plan', fontsize=fs)
ax[1].set_ylabel(r'($\mathbf{x}_1, \cdots, \mathbf{x}_n$)')
x_label = r'($\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_1, \cdots,'
x_label += r'\mathbf{U}\mathbf{U}^{\top}\mathbf{x}_n$)'
ax[1].set_xlabel(x_label)

plt.tight_layout()
save_figure(folder_path, 'example_gmm')
