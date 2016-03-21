from __future__ import division
import numpy as np
import numpy.random as npr
import cPickle as pickle
import gzip
import h5py
import operator as op


partial_flatten = lambda x: np.reshape(x, (x.shape[0], -1))

dmap = lambda dct, f=lambda x: x, keep=lambda x: True: \
    {k:f(v) for k, v in dct.iteritems() if keep(k)}

def standardize(d):
    recenter = lambda d: d - np.percentile(d, 0.01)
    rescale = lambda d: d / np.percentile(d, 99.99)
    return rescale(recenter(d))

def flatten_dict(dct):
    data = map(op.itemgetter(1), sorted(dct.items(), key=op.itemgetter(0)))
    return np.concatenate(data)

def load(filename):
    openfile = open if filename.endswith('.pkl') else gzip.open
    with openfile(filename, 'r') as infile:
        datadict = pickle.load(infile)
    return datadict

def load_mice(N, file, labelfile=None, addnoise=True, keep=lambda x: True):
    print 'loading data from {}...'.format(file)
    if labelfile is None:
        data = _load_mice(N, file, keep)
    else:
        data, labels = _load_mice_withlabels(N, file, labelfile, keep)

    if addnoise:
        data += 1e-3 * npr.normal(size=data.shape)

    print '...done loading {} frames!'.format(len(data))
    if labelfile:
        return data, labels
    return data

def _load_mice(N, file, keep):
    datadict = dmap(load(filename), standardize, keep)
    return flatten_dict(datadict)

def _load_mice_withlabels(N, file, labelfile, keep):
    datadict = dmap(load(file), standardize, keep)

    with open(labelfile, 'r') as infile:
        stateseqs_dict = dmap(pickle.load(infile), keep=keep)

    def truncate(a, b):
        l = min(len(a), len(b))
        return a[-l:], b[-l:]

    merged_dict = {name: truncate(datadict[name], stateseqs_dict[name][-1])
                    for name in stateseqs_dict}
    pairs = map(op.itemgetter(1), sorted(merged_dict.items(), key=op.itemgetter(0)))
    data, labels = map(np.concatenate, zip(*pairs))
    data, labels = partial_flatten(data[:N]), labels[:N]

    _, labels = np.unique(labels, return_inverse=True)

    return data, labels

def load_vae_init(zdim, file, eps=1e-5):
    with gzip.open(file, 'r') as infile:
        encoder_params, decoder_params = pickle.load(infile)

    encoder_nnet_params, ((W_h, b_h), (W_J, b_J)) = encoder_params[:-2], encoder_params[-2:]
    (W_1, b_1), decoder_nnet_params = decoder_params[0], decoder_params[1:]

    if zdim < W_h.shape[1]:
        raise ValueError, 'initialization zdim must not be greater than svae model zdim'
    elif zdim > W_h.shape[1]:
        padsize = zdim - W_h.shape[1]
        pad = lambda W, b: \
            (np.hstack((eps*npr.randn(W.shape[0], padsize), W)),
            np.concatenate((eps*npr.randn(padsize), b)))
        encoder_params = encoder_nnet_params + [pad(W_h, b_h), pad(W_J, b_J)]

        # pad out decoder weights to match zdim
        pad = lambda W, b: (np.vstack((eps*npr.randn(padsize, W.shape[1]), W)), b)
        decoder_params = [pad(W_1, b_1)] + decoder_nnet_params

        print 'loaded init from {} and padded by {} dimensions'.format(file, padsize)
        return encoder_params, decoder_params

    print 'loaded init from {}'.format(file)
    return encoder_params, decoder_params
