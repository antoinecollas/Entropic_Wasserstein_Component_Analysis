import numpy as np
from numpy import linalg as la
import pandas
import scipy.io as sio


def check_dataset(dataset):
    datasets = ['khan2001', 'Breast']
    assert dataset in datasets, f'Dataset {dataset} not in {datasets}'
    if dataset == 'khan2001':
        return load_khan2001
    if dataset == 'Breast':
        return load_Breast


def load_khan2001(
    path_dataset='datasets/khan2001.mat',
    verbose=True
):
    # load and center data
    matstruct_contents = sio.loadmat(path_dataset)
    X = matstruct_contents['Xo'].T
    X = X - X.mean(0)
    y = matstruct_contents['yo']
    y = y.reshape(-1)

    # check shape
    assert X.shape == (63, 2308)
    assert la.norm(np.mean(X, axis=0)) <= 1e-6
    assert len(y) == 63
    assert len(np.unique(y)) == 4

    # sort data by classes numbers
    idx = np.argsort(y)
    X = X[idx, :]
    y = y[idx]

    # print infos
    if verbose:
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        for i in np.unique(y):
            print('Class '+str(i)+': '+str(np.sum(y == i))+' samples.')
        print()

    return X, y


def load_Breast(
    path_dataset='datasets/Breast_GSE45827.csv',
    verbose=True
):
    def converter(s):
        if s == 'HER':
            return 0
        if s == 'basal':
            return 1
        if s == 'cell_line':
            return 2
        if s == 'luminal_A':
            return 3
        if s == 'luminal_B':
            return 4
        if s == 'normal':
            return 5

    converters = {'type': converter}
    data = pandas.read_csv(path_dataset, converters=converters)
    y = data['type'].values
    data = data.drop(['samples', 'type'], axis='columns')
    X = data.values
    X = X - X.mean(0)
    assert X.shape == (151, 54675)
    assert X.dtype == np.float64
    assert len(np.unique(y)) == 6
    assert y.dtype == np.int64

    # sort data by classes numbers
    idx = np.argsort(y)
    X = X[idx, :]
    y = y[idx]

    # print infos
    if verbose:
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        for i in np.unique(y):
            print('Class '+str(i)+': '+str(np.sum(y == i))+' samples.')
        print()

    return X, y
