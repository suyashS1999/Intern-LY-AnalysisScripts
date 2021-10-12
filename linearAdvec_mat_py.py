import numpy as np

def Adv_mat(N):
	A = np.zeros((N, N));
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

def Diff_mat(N):
	D = np.zeros((N, N));
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

def Compute_eigVals(N, alpha, kdt):
	A = Adv_mat(N);
	D = Diff_mat(N);
	ADt = A*(-alpha/2.0) + D*kdt;
	ldt = np.zeros(N, 'complex');
	beta = np.zeros(N);
	for m in range(N):
		beta[m] = 2*np.pi*m/N;
		if beta[m] > np.pi:
			beta[m] = 2*np.pi - beta[m];
		for j in range(N):
			ldt[m] += ADt[0, j]*np.exp(1j*2.0*np.pi*j*m/N);
	return ldt, beta;