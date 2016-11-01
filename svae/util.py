from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.linalg as spla
from autograd.util import flatten
from itertools import islice, imap, cycle
import operator
from functools import partial
from toolz import curry

# autograd internals
from autograd.container_types import TupleNode, ListNode
from autograd.core import getval, primitive


### neural nets

identity = lambda x: x
sigmoid = lambda x: 1. / (1. + np.exp(-x))
relu = lambda x: np.maximum(x, 0.)
softplus = partial(np.logaddexp, 0.)
normalize = lambda x: x / np.sum(x, axis=-1, keepdims=True)
softmax = lambda x: normalize(np.exp(x - np.max(x, axis=-1, keepdims=True)))


### misc

def rle(stateseq):
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)

isarray = lambda x: hasattr(x, 'ndim')
flat = lambda x: flatten(x)[0]
partial_flat = lambda a, axes: np.reshape(a, a.shape[:-axes] + (-1,))
tensordot = lambda a, b, axes=2: np.dot(partial_flat(a, axes), partial_flat(b, axes).T)
outer = lambda x, y: x[...,:,None] * y[...,None,:]

### functions and monads

def compose(funcs):
    def composition(x):
        for f in funcs:
            x = f(x)
        return x
    return composition

def monad_runner(bind):
    def run(result, steps):
        for i, step in enumerate(steps):
            result = bind(result, step)
        return result
    return run


### matrices

T = lambda X: np.swapaxes(X, axis1=-1, axis2=-2)
symmetrize = lambda X: (X + T(X))/2.

def solve_triangular(L, x, trans='N'):
    return spla.solve_triangular(L, x, lower=True, trans=trans)

def solve_posdef_from_cholesky(L, x):
    return solve_triangular(L, solve_triangular(L, x), 'T')

def solve_symmetric(A, b):
    return np.linalg.solve(symmetrize(A), b)

def rand_psd(n):
    A = npr.randn(n, n)
    return np.dot(A, A.T)

def rot2D(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])



### lists

def interleave(a, b):
    return list(roundrobin(a, b))

def uninterleave(lst):
    return lst[::2], lst[1::2]

def roundrobin(*iterables):
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


### splitting data into batches

def get_num_datapoints(x):
    return x.shape[0] if isinstance(x, np.ndarray) else sum(map(get_num_datapoints, x))

def flatmap(f, container):
    flatten = lambda lst: [item for sublst in lst for item in sublst]
    mappers = {np.ndarray: lambda f, arr: f(arr),
                     list: lambda f, lst: flatten(map(f, lst)),
                     dict: lambda f, dct: flatten(map(f, dct.values()))}
    return mappers[type(container)](f, container)

@curry
def split_array(arr, length):
    truncate_to_multiple = lambda arr, k: arr[:k*(len(arr) // k)]
    return np.split(truncate_to_multiple(arr, length), len(arr) // length)

def split_into_batches(data, seq_len, num_seqs=None, permute=True):
    batches = npr.permutation(flatmap(split_array(length=seq_len), data))
    if num_seqs is None:
        return batches, len(batches)
    chunks = (batches[i*num_seqs:(i+1)*num_seqs] for i in xrange(len(batches) // num_seqs))
    return imap(np.stack, chunks), len(batches) // num_seqs


### basic math on (nested) tuples

istuple = lambda x: isinstance(x, (tuple, TupleNode, list, ListNode))
ensuretuple = lambda x: x if istuple(x) else (x,)
concat = lambda *args: reduce(operator.add, map(ensuretuple, args))
inner = lambda a, b: np.dot(np.ravel(a), np.ravel(b))

Y = lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args)))
make_unop = lambda op, combine: \
    Y(lambda f: lambda a: op(a) if not istuple(a) else combine(map(f, a)))
make_scalar_op = lambda op, combine: \
    Y(lambda f: lambda a, b : op(a, b)  if not istuple(b) else combine(map(partial(f, a), b)))
make_binop = lambda op, combine: \
    Y(lambda f: lambda a, b: op(a, b) if not istuple(a) else combine(map(f, a, b)))

def add_binop_size_check(binop):
    def wrapped(a, b):
        assert shape(a) == shape(b)
        return binop(a, b)
    return wrapped
make_binop = (lambda make_binop: lambda *args:
              add_binop_size_check(make_binop(*args)))(make_binop)

add      = make_binop(operator.add,     tuple)
sub      = make_binop(operator.sub,     tuple)
mul      = make_binop(operator.mul,     tuple)
div      = make_binop(operator.truediv, tuple)
allclose = make_binop(np.allclose,      all)
contract = make_binop(inner,            sum)

shape      = make_unop(np.shape, tuple)
unbox      = make_unop(getval,   tuple)
sqrt       = make_unop(np.sqrt,  tuple)
square     = make_unop(lambda a: a**2, tuple)
randn_like = make_unop(lambda a: npr.normal(size=np.shape(a)), tuple)
zeros_like = make_unop(lambda a: np.zeros(np.shape(a)), tuple)
flatten    = make_unop(lambda a: np.ravel(a), np.concatenate)

scale      = make_scalar_op(operator.mul, tuple)
add_scalar = make_scalar_op(operator.add, tuple)

norm = lambda x: np.sqrt(contract(x, x))
rand_dir_like = lambda x: scale(1./norm(x), randn_like(x))

isobjarray = lambda x: isinstance(x, np.ndarray) and x.dtype == np.object
tuplify = Y(lambda f: lambda a: a if not istuple(a) and not isobjarray(a) else tuple(map(f, a)))
depth = Y(lambda f: lambda a: np.ndim(a) if not istuple(a) else 1+(min(map(f, a)) if len(a) else 1))
