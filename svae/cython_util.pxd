# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

from scipy.linalg.cython_blas cimport dsymv, ddot, dgemm, dgemv
from scipy.linalg.cython_lapack cimport dtrtrs, dpotrf, dpotrs, dpotri

cdef inline void copy(double[::1,:] a, double[::1,:] b) nogil:
    cdef int M = a.shape[0], N = a.shape[1]
    cdef int i, j

    for j in range(N):
        for i in range(M):
            b[i,j] = a[i,j]

cdef inline void copy_vector(double[::1] a, double[::1] b) nogil:
    cdef int M = a.shape[0]
    cdef int i

    for i in range(M):
        b[i] = a[i]

cdef inline void zero_matrix(double[::1,:] a) nogil:
    cdef int M = a.shape[0], N = a.shape[1]
    cdef int i, j

    for j in range(N):
        for i in range(M):
            a[i,j] = 0.

cdef inline void copy_lower_to_upper(double[::1,:] a) nogil:
    cdef int i, j, N = a.shape[0]

    for j in range(N):
        for i in range(j, N):
            a[j,i] = a[i,j]


cdef inline void zero_vector(double[::1] a) nogil:
    cdef int M = a.shape[0]
    cdef int i

    for i in range(M):
        a[i] = 0.

cdef inline void zero_upper_triangle(double[::1,:] a) nogil:
    cdef int M = a.shape[0], N = a.shape[1]
    cdef int i, j

    for j in range(N):
        for i in range(j):
            a[i,j] = 0.

cdef inline void add_into(double[::1,:] a, double[::1,:] b) nogil:
    cdef int M = a.shape[0], N = a.shape[1]
    cdef int i, j

    for j in range(N):
        for i in range(M):
            b[i,j] += a[i,j]

cdef inline void add_into_vector(double[::1] a, double[::1] b) nogil:
    cdef int M = a.shape[0]
    cdef int i

    for i in range(M):
        b[i] += a[i]

cdef inline void symmetrize(double[::1,:] a) nogil:
    cdef int M = a.shape[0]
    cdef int i, j

    for j in range(M):
        for i in range(M):
            a[i,j] = a[j,i] = (a[i,j] + a[j,i])/2.

cdef inline void scale(double scalar, double[::1,:] a) nogil:
    cdef int M = a.shape[0], N = a.shape[1]
    cdef int i, j

    for j in range(N):
        for i in range(M):
            a[i,j] *= scalar

cdef inline double max_vector(double[::1] a) nogil:
    cdef int i, N = a.shape[0]
    cdef double themax

    themax = a[0]
    for i in range(1, N):
        themax = max(themax, a[i])

    return themax

cdef inline void transpose(double[::1,:] a) nogil:
    cdef int i, j, N = a.shape[0]

    for j in range(N):
        for i in range(j):
            a[i,j], a[j,i] = a[j,i], a[i,j]
