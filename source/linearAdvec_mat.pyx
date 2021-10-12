import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport pi
cdef extern from "complex.h":
	double complex cexp(double complex);

DTYPE = np.float64;
DTYPE_c = np.complex128;
ctypedef np.float64_t DTYPE_t;

cdef np.ndarray[DTYPE_t, ndim = 2] Adv_mat(int N):
	cdef np.ndarray[DTYPE_t, ndim = 2] A = np.zeros((N, N), dtype = DTYPE);
	cdef int i;
	for i in range(N):
		if i == 0:
			A[i, N - 1] = -1.0;
			A[i, i] = 0.0;
			A[i, i + 1] = 1.0;
		elif i == N - 1:
			A[i, i - 1] = -1.0;
			A[i, i] = 0.0;
			A[i, 0] = 1.0;
		else:
			A[i, i - 1] = -1.0;
			A[i, i] = 0.0;
			A[i, i + 1] = 1.0;
	return A;

cdef np.ndarray[DTYPE_t, ndim = 2] Diff_mat(int N):
	cdef np.ndarray[DTYPE_t, ndim = 2] D = np.zeros((N, N), dtype = DTYPE);
	cdef int i;
	for i in range(N):
		if i == 0:
			D[i, N - 1] = 1.0;
			D[i, i] = -2.0;
			D[i, i + 1] = 1.0;
		elif i == N - 1:
			D[i, i - 1] = 1.0;
			D[i, i] = -2.0;
			D[i, 0] = 1.0;
		else:
			D[i, i - 1] = 1.0;
			D[i, i] = -2.0;
			D[i, i + 1] = 1.0;
	return D;

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t, ndim = 1] Compute_eigVals(int N, double alpha, double kdt):
	cdef np.ndarray[DTYPE_t, ndim = 2] A = Adv_mat(N);
	cdef np.ndarray[DTYPE_t, ndim = 2] D = Diff_mat(N);
	cdef np.ndarray[np.complex128_t, ndim = 2] ADt = A*(-alpha/2.0) + D*kdt + 0j;
	cdef np.ndarray[np.complex128_t, ndim = 1] ldt = np.zeros(N, dtype = DTYPE_c);
	cdef np.ndarray[DTYPE_t, ndim = 1] beta = np.zeros(N, dtype = DTYPE);
	cdef int m, k;
	cdef double two_pi = 2*pi;
	cdef double complex factor = 1j*2.0*pi + 0;
	for m in range(N):
		beta[m] = two_pi*m/N;
		if beta[m] > pi:
			beta[m] = two_pi - beta[m];
		for k in range(N):
			ldt[m] = ldt[m] + ADt[0, k]*cexp(factor*k*m/N);
	return ldt;
