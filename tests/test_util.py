from __future__ import division
import numpy as np
import numpy.random as npr


def info_to_mean(infoparams):
    J, h = infoparams
    return np.linalg.solve(J, h), np.linalg.inv(J)


def mean_to_info(mu, Sigma):
    return np.linalg.inv(Sigma), np.linalg.solve(Sigma, mu)


def natural_to_mean(natparams):
    neghalfJ, h = natparams
    return np.linalg.solve(-2*neghalfJ, h), np.linalg.inv(-2*neghalfJ)


def rand_psd(n):
    temp = npr.randn(n,n)
    return np.dot(temp, temp.T)


def cumsum(v,strict=False):
    if not strict:
        return np.cumsum(v,axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1],axis=0)
        return out


def bmat(blocks):
    rowsizes = [row[0].shape[0] for row in blocks]
    colsizes = [col[0].shape[1] for col in zip(*blocks)]
    rowstarts = cumsum(rowsizes,strict=True)
    colstarts = cumsum(colsizes,strict=True)

    nrows, ncols = sum(rowsizes), sum(colsizes)
    out = np.zeros((nrows,ncols))

    for i, (rstart, rsz) in enumerate(zip(rowstarts, rowsizes)):
        for j, (cstart, csz) in enumerate(zip(colstarts, colsizes)):
            out[rstart:rstart+rsz,cstart:cstart+csz] = blocks[i][j]

    return out
