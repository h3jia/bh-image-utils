import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
# from libc.stdio cimport printf
# from cython.parallel import prange, parallel

# TODO: use fused type

__all__ = ['_ri_theta', '_ri_io']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _find_max_double(const double *f, const long n) nogil except -1:
    cdef size_t i, m=0
    for i in range(1, n):
        if f[i] > f[m]:
            m = i
    return m


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _find_max_long(const long *f, const long n) nogil except -1:
    cdef size_t i, m=0
    for i in range(1, n):
        if f[i] > f[m]:
            m = i
    return m


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _find_min_double(const double *f, const long n) nogil except -1:
    cdef size_t i, m=0
    for i in range(1, n):
        if f[i] < f[m]:
            m = i
    return m


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _find_min_long(const long *f, const long n) nogil except -1:
    cdef size_t i, m=0
    for i in range(1, n):
        if f[i] < f[m]:
            m = i
    return m


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _ri_theta_1(const double *disk_f, long *ri_theta, long *is_bad,
                     const long n_r, const double alpha, const double beta) nogil except -1:
    cdef long n_peak=0, m, i_m, i_in, i_out
    cdef size_t i, j
    cdef double f_m
    cdef long *i_peak = <long *> malloc(n_r * sizeof(long))
    cdef double *f_peak = <double *> malloc(n_r * sizeof(double))
    for i in range(1, n_r - 1):
        if disk_f[i] > disk_f[i - 1] and disk_f[i] > disk_f[i + 1]:
            i_peak[n_peak] = i
            f_peak[n_peak] = disk_f[i]
            n_peak += 1
    if not n_peak:
        free(i_peak)
        free(f_peak)
        return 1
    else:
        m = _find_max_double(f_peak, n_peak)
        f_m = f_peak[m]
        i_m = i_peak[m]
        j = 0
        while j < n_peak:
            if f_peak[j] < f_m * alpha:
                f_peak[j] = f_peak[n_peak - 1]
                i_peak[j] = i_peak[n_peak - 1]
                n_peak -= 1
            else:
                j += 1
        i_out = i_peak[_find_max_long(i_peak, n_peak)]
        i_in = i_peak[_find_min_long(i_peak, n_peak)]
        f_mm = f_peak[_find_min_double(f_peak, n_peak)]
        ri_theta[0] = i_m
        for j in range(i_in, i_out + 1):
            if disk_f[j] < f_mm * beta:
                is_bad[0] = j
                break
        if disk_f[0] > f_m or disk_f[n_r - 1] > f_m:
            is_bad[0] = -1
        free(i_peak)
        free(f_peak)
        return 0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _ri_theta(const double[:, ::1] disk_f, long[::1] ri_theta, long[::1] is_bad,
              const long n_theta, const long n_r, const double alpha,
              const double beta):
    cdef size_t i
    for i in range(n_theta):
        if _ri_theta_1(&disk_f[i, 0], &ri_theta[i], &is_bad[i], n_r, alpha, beta):
            is_bad[i] = -2


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef long _ri_io_1(const double *disk_f, const long ri_theta, long *ri_in, long *ri_out,
                   const long n_r, const double f_theta) nogil except -1:
    cdef size_t i = 0
    while disk_f[i] < f_theta:
        if i < n_r:
            i += 1
        else:
            return 1
    ri_in[0] = i
    i = n_r - 1
    while disk_f[i] < f_theta:
        if i > 0:
            i -= 1
        else:
            return 1
    ri_out[0] = i
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _ri_io(const double[:, ::1] disk_f, const long[::1] ri_theta, long[::1] is_bad,
           long[::1] ri_in, long[::1] ri_out, const long n_theta, const long n_r,
           const double[::1] f_theta):
    cdef size_t i
    for i in range(n_theta):
        if is_bad[i]:
            ri_in[i] = -1
            ri_out[i] = -1
        else:
            if _ri_io_1(&disk_f[i, 0], ri_theta[i], &ri_in[i], &ri_out[i], n_r, f_theta[i]):
                ri_in[i] = -1
                ri_out[i] = -1
                is_bad[i] = -3
