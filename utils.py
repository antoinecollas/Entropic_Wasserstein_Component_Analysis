from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


# good layout for double columns signal processing paper
# font size, figure size, ...
plt.rcParams.update({'pdf.fonttype': 42})
plt.rcParams.update({'mathtext.fontset': 'cm'})  # create weird messages
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': (6, 4)})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'lines.markersize': 6})
plt.rcParams.update({'legend.fontsize': 9})
plt.rcParams.update({'axes.labelsize': 12})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'savefig.dpi': 600})
plt.rcParams.update({'savefig.bbox': 'tight'})
plt.rcParams.update({'savefig.pad_inches': 0.1})
plt.rcParams.update({'savefig.transparent': True})
plt.rcParams.update({'savefig.facecolor': 'w'})
plt.rcParams.update({'savefig.edgecolor': 'w'})
plt.rcParams.update({'savefig.orientation': 'portrait'})


def create_directory(name):
    """ A function that creates a directory to save results.
    The path has the 'results/name/date'.
    """
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path = Path('figures')
    path /= name
    path /= date_str
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(folder_path, figname):
    """ A function that save the current figure in '.pdf'.
        Inputs:
            * folder_path: path of the folder where to save actual figure.
            * figname: string of the name of the figure to save.
    """
    p = folder_path

    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    path = p / figname
    path_pdf = path.with_suffix('.pdf')

    # save the figure
    plt.savefig(path_pdf)
