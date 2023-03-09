import geoopt
import numpy as np
import ot
import scipy
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import warnings
import time


class NanError(Exception):
    def __init__(self, message):
        self.message = message


def compute_loss(M, G, reg):
    if type(M) == type(G) == torch.Tensor:
        return torch.sum(M*G) + reg*torch.sum(G*(torch.log(G)-1))
    if type(M) == type(G) == np.ndarray:
        return np.sum(M*G) + reg*np.sum(G*(np.log(G)-1))
    return TypeError('Error: should be np.ndarray or torch.Tensor...')


def subspace_relative_distance(P1, P2):
    # slow when the ambient dimension is large
    proj1 = P1@P1.T
    proj2 = P2@P2.T
    if type(P1) == type(P2) == torch.Tensor:
        return torch.linalg.norm(proj1-proj2)/torch.linalg.norm(proj1)
    if type(P1) == type(P2) == np.ndarray:
        return np.linalg.norm(proj1-proj2)/np.linalg.norm(proj1)
    return TypeError('Error: should be np.ndarray or torch.Tensor...')


def Grassmann_distance(P1, P2):
    proj = P1.T@P2
    if type(P1) == type(P2) == torch.Tensor:
        _, s, _ = torch.linalg.svd(proj)
        s[s > 1] = 1
        s = torch.arccos(s)
        return torch.linalg.norm(s)
    if type(P1) == type(P2) == np.ndarray:
        _, s, _ = np.linalg.svd(proj)
        s[s > 1] = 1
        s = np.arccos(s)
        return np.linalg.norm(s)
    return TypeError('Error: should be np.ndarray or torch.Tensor...')


class StiefelLinear(torch.nn.Module):
    def __init__(self, k, d, device):
        super(StiefelLinear, self).__init__()
        self.d = d
        self.k = k
        self.device = device
        self.weight = geoopt.ManifoldParameter(
            data=torch.Tensor(d, k),
            manifold=geoopt.Stiefel(canonical=False)
        )
        self._init_done = False

    def reset_parameters(self, input):
        with torch.no_grad():
            U, _, _ = torch.linalg.svd(input.T, full_matrices=False)
            self.weight.data = U[:, :self.k]
        self.weight.data = self.weight.data.to(self.device)  # TODO: necessary?
        self._init_done = True

    def forward(self, input):
        if self._init_done is False:
            self.reset_parameters(input)
        return F.linear(input, self.weight)

    def forward_T(self, input):
        if self._init_done is False:
            self.reset_parameters(input)
        return F.linear(input, self.weight.T)

    def fforward(self, input):
        if self._init_done is False:
            self.reset_parameters(input)
        return F.linear(input, self.weight @ self.weight.T)


def ot_pca_bcd(
    X, k, reg=1e-3, thresh=1e-4, max_iter=10000,
    P0=None, max_iter_sink=100, max_iter_MM=10,
    verbose=False, log=True, method_sink='sinkhorn',
    method='MM', svd_fct_cpu='numpy', device='cpu',
    warn=True
):
    if verbose is False:
        verbose = 0
    elif verbose is True:
        verbose = 1

    if verbose > 0:
        print()

    assert device in ['cpu', 'cuda']

    if device == 'cuda':
        X = torch.tensor(X, dtype=torch.float64)
        X = X.to(device)

    n, d = X.shape
    X = X - X.mean(0)  # center data
    assert k <= d

    t0 = time.time()
    if P0 is None:
        if device == 'cpu':
            pca_fitted = PCA(n_components=k).fit(X)
            P = pca_fitted.components_.T
            if method == 'MM':
                lambda_scm = pca_fitted.explained_variance_[0]
        elif device == 'cuda':
            P, s, _ = torch.linalg.svd(X.T, full_matrices=False)
            P = P[:, :k]
            if method == 'MM':
                lambda_scm = (1/n)*(s[0]**2)
    else:
        P = P0
    if verbose > 1:
        print('init time:', time.time() - t0)

    # marginals
    if device == 'cpu':
        u0 = (1./n)*np.ones(n)
    elif device == 'cuda':
        u0 = (1./n)*torch.ones(n, dtype=torch.float64).to(device)

    # log
    if log:
        log_ = {}
        log_['P'] = []
        log_['G'] = []
        if device == 'cpu':
            log_['P0'] = P
        elif device == 'gpu':
            log_['P0'] = P.detach().clone().cpu().numpy()
        log_['loss'] = []

    # print iterations
    if verbose > 0:
        if method == 'BCD':
            print('OTPCA BCD')
        elif method == 'MM':
            print('OTPCA MM')
        else:
            raise ValueError('Unknown method...')
        print('{:4s}|{:13s}|{:12s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Crit.', 'Thres.', 'Time.'
        ) + '\n' + '-' * 48)

    # loop
    go = True
    it = 0
    warmstart_sinkhorn = None

    while go:
        st = time.time()
        P_old = P

        # Solve transport
        t0 = time.time()
        M = ot.dist(X, (X@P)@P.T)
        G, log_sinkhorn = ot.sinkhorn(
            u0, u0, M, reg,
            numItermax=max_iter_sink,
            method=method_sink, warmstart=warmstart_sinkhorn,
            warn=warn, log=True
        )
        key_warmstart = 'warmstart'
        if key_warmstart in log_sinkhorn:
            alpha, beta = log_sinkhorn[key_warmstart]
            if type(alpha) == type(beta) == torch.Tensor:
                alpha, beta = alpha.detach(), beta.detach()
            warmstart_sinkhorn = (alpha, beta)
        if (G >= 1e-300).all():
            loss = compute_loss(M, G, reg)
        else:
            if device == 'cpu':
                loss = np.inf
            elif device == 'cuda':
                loss = torch.inf
        if verbose > 1:
            print('OT time:', time.time() - t0)

        # log
        if log:
            if device == 'cpu':
                log_['P'].append(P)
                log_['G'].append(G)
                log_['loss'].append(loss)
            elif device == 'cuda':
                log_['P'].append(P.detach().clone().cpu().numpy())
                log_['G'].append(G.detach().clone().cpu().numpy())
                log_['loss'].append(loss.item())

        # Solve PCA
        t0 = time.time()
        G_sym = (G+G.T)/2

        if method == 'BCD':
            # block coordinate descent
            t0_bcd = time.time()

            if device == 'cpu':
                eye = np.eye(n)
            elif device == 'cuda':
                eye = torch.eye(n).to(device)
            S = X.T @ (2*G_sym-(1./n)*eye) @ X

            if verbose > 1:
                print('BCD S computation time:', time.time() - t0_bcd)
            t1_bcd = time.time()

            if device == 'cpu':
                if svd_fct_cpu == 'scipy':
                    _, U = scipy.sparse.linalg.eigsh(S, k, which='LA')
                    P = U[:, ::-1]
                elif svd_fct_cpu == 'numpy':
                    _, U = np.linalg.eigh(S)
                    P = U[:, ::-1][:, :k]
                else:
                    raise ValueError('Wrong value for svd_fct_cpu...')
            elif device == 'cuda':
                _, U = torch.linalg.eigh(S)
                idx = np.array(np.arange(d)[::-1][:k])
                P = U[:, idx]

            if verbose > 1:
                print('BCD svd computation time:', time.time() - t1_bcd)

        elif method == 'MM':
            # majorization-minimization
            t0_mm = time.time()

            if device == 'cpu':
                if svd_fct_cpu == 'scipy':
                    eig, _ = scipy.sparse.linalg.eigsh(G_sym, k=1, which='SA')
                elif svd_fct_cpu == 'numpy':
                    eig, _ = np.linalg.eigh(G_sym)
                else:
                    raise ValueError('Wrong value for svd_fct_cpu...')
                lambda_G = eig[0]
            elif device == 'cuda':
                raise NotImplementedError

            if verbose > 1:
                print('MM lambda min computation time:', time.time() - t0_mm)

            t_R, t_proj = 0, 0
            for _ in range(max_iter_MM):
                t1_mm = time.time()

                X_proj = X@P
                X_T_X_proj = X.T@X_proj

                R = (1/n) * X_T_X_proj
                alpha = 1 - 2*n*lambda_G
                if alpha > 0:
                    R = alpha*(R - lambda_scm*P)
                else:
                    R = alpha*R

                R = R - (2*X.T@(G_sym@X_proj)) + (2*lambda_G*X_T_X_proj)

                t_R += time.time() - t1_mm
                t2_mm = time.time()

                if device == 'cpu':
                    P, _ = np.linalg.qr(R)
                    # U, s, Vt = np.linalg.svd(R, full_matrices=False)
                    # P = U@Vt
                elif device == 'cuda':
                    P, _ = torch.linalg.qr(R)

                t_proj += time.time() - t2_mm

            if verbose > 1:
                print('MM R computation time:', t_R)
                print('MM proj computation time:', t_proj)

        else:
            raise ValueError('Unknown method...')

        if verbose > 1:
            print('Update U time:', time.time() - t0)

        # stop or not
        crit = Grassmann_distance(P_old, P)
        if crit <= thresh:
            go = False
        it += 1
        if it >= max_iter:
            go = False
            if warn:
                warnings.warn('ot_pca_bcd: max iter reached')

        # print
        ed = time.time()
        if verbose > 0:
            print('{:4d}|{:8e}|{:8e}|{:8e}|{:8e}'.format(
                it, loss, crit, thresh, ed-st))

    if log:
        log_['it'] = it

    if device == 'cuda':
        G = G.detach().clone().cpu().numpy()
        P = P.detach().clone().cpu().numpy()

    if log:
        return G, P, log_
    else:
        return G, P


def ot_pca_auto_diff(
    X, k, reg=1e-3, lr=1e-2, thresh=1e-3, max_iter=100,
    P0=None, max_iter_sink=100, verbose=False, log=True,
    method_sink='sinkhorn', device=None, warn=True
):
    if verbose is False:
        verbose = 0
    elif verbose is True:
        verbose = 1

    if verbose > 0:
        print()

    # optimization on the Stiefel manifold
    manifold = geoopt.Stiefel()

    # X
    X = torch.from_numpy(X).type(torch.float64).to(device)
    X = X - X.mean(0)  # center data
    n, d = X.shape
    assert k <= d

    # device
    if device is None:
        device = torch.device('cpu')
    else:
        device = device

    # init subspace basis
    if P0 is None:
        P = StiefelLinear(d=d, k=k, device=device)
        P.reset_parameters(X)
    else:
        P = P0

    # marginals
    u0 = (1./n)*torch.ones(n, dtype=torch.float64).to(device)

    # optimizer
    optimizer = geoopt.optim.RiemannianAdam(P.parameters(), lr=lr)

    # log
    if log:
        log_ = {}
        log_['P'] = []
        log_['G'] = []
        log_['P0'] = P.weight.detach().clone().numpy()
        log_['loss'] = []
        log_['gradnorm'] = []
        log_['backward'] = []

    # print iterations
    if verbose:
        print('OTPCA Riemannian')
        print('{:4s}|{:13s}|{:12s}|{:12s}|{:8s}'.format(
            'It.', 'Loss', 'Crit.', 'Thres.', 'Time.'
        ) + '\n' + '-' * 48)

    # loop
    go = True
    it = 0
    warmstart_sinkhorn = None

    while go:
        st = time.time()
        optimizer.zero_grad()
        P_old = P.weight.data.detach().clone()

        # compute loss
        t0 = time.time()
        M = ot.dist(X, P.forward(P.forward_T(X)))
        G, log_sinkhorn = ot.sinkhorn(
            u0, u0, M, reg,
            numItermax=max_iter_sink,
            method=method_sink, warmstart=warmstart_sinkhorn,
            warn=warn, log=True
        )
        key_warmstart = 'warmstart'
        if key_warmstart in log_sinkhorn:
            alpha, beta = log_sinkhorn[key_warmstart]
            if type(alpha) == type(beta) == torch.Tensor:
                alpha, beta = alpha.detach(), beta.detach()
            warmstart_sinkhorn = (alpha, beta)
        loss = compute_loss(M, G, reg)
        if torch.isnan(loss).any():
            print(G)
            print(M)
            raise NanError('Loss is Nan')
        if verbose > 1:
            print(f'Loss evaluation time: {time.time() - t0:,.4f}s')

        # gradient step
        t0 = time.time()
        loss.backward()
        if verbose > 1:
            print(f'Gradient evaluation time: {time.time() - t0:,.4f}s')
        t0 = time.time()
        optimizer.step()
        t_backward = time.time() - t0
        if verbose > 1:
            print(f'Optimizer step time: {time.time() - t0:,.4f}s')

        # draft of a possibly fater gradient computation
        # X_proj = P.forward_T(X)
        # loss = - torch.sum(u0 * (torch.linalg.norm(X_proj, axis=1)**2))
        # gram_matrix = X_proj @ X_proj.T
        # loss = loss - 2 * torch.sum(G*gram_matrix)
        # loss.backward()

        # log
        t0 = time.time()
        if log:
            log_['backward'].append(t_backward)
            log_['P'].append(P.weight.data.detach().clone().numpy())
            log_['G'].append(G.data.detach().clone().numpy())
            log_['loss'].append(loss.item())
            rgrad = manifold.egrad2rgrad(P.weight, P.weight.grad)
            gradnorm = manifold.inner(P.weight, rgrad).item()
            log_['gradnorm'].append(gradnorm)
        if verbose > 1:
            print(f'Log time: {time.time() - t0:,.4f}s')

        # stop conditions
        t0 = time.time()
        crit = Grassmann_distance(P_old, P.weight.data)
        it += 1
        if crit <= thresh:
            go = False
        if it >= max_iter:
            go = False
        if verbose > 1:
            print(f'Stopping criteria eval. time: {time.time() - t0:,.4f}s')

        # print
        ed = time.time()
        if verbose:
            print('{:4d}|{:8e}|{:8e}|{:8e}|{:8e}'.format(
                it, loss, crit, thresh, ed-st))

    P = P.weight.data.detach().clone().numpy()
    G = G.data.detach().clone().numpy()

    if log:
        log_['it'] = it
        return G, P, log_
    else:
        return G, P
