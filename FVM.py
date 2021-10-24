import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import functools
import matplotlib.pyplot as plt

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Gauss_linear(N):
	A = sp.diags(0.5*np.ones(N - 1, dtype = np.float64), offsets = 1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(-0.5*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64);
	A += sp.csr_matrix(([-0.5, 0.5], ([0, N - 1], [N - 1, 0])), shape = (N, N));
	return A;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Gauss_linear_with_artificial_diffusion(N, k = 0.1):
	A = sp.diags(0.5*np.ones(N - 1, dtype = np.float64), offsets = 1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(-0.5*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64);
	A += sp.csr_matrix(([-0.5, 0.5], ([0, N - 1], [N - 1, 0])), shape = (N, N));
	A -= k*Diff_mat_Gauss_linear(N);
	return A;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Gauss_upwind(N):
	A = sp.diags(np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(-1*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64);
	A += sp.csr_matrix(([-1], ([0], [N - 1])), shape = (N, N));
	return A;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Gauss_linearUpwind(N):
	A = sp.diags(1.5*np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(-2.0*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(0.5*np.ones(N - 2, dtype = np.float64), offsets = -2, shape = (N, N), format = 'csr', dtype = np.float64);
	A += sp.csr_matrix(([-2.0, 0.5, 0.5], ([0, 0, 1], [N - 1, N - 2, N - 1])), shape = (N, N));
	# A = sp.diags(0.75*np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64) \
	#   + sp.diags(-1.25*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64) \
	#   + sp.diags(0.25*np.ones(N - 2, dtype = np.float64), offsets = -2, shape = (N, N), format = 'csr', dtype = np.float64) \
	#   + sp.diags(0.25*np.ones(N - 1, dtype = nCrank_Nicolsonp.float64), offsets = 1, shape = (N, N), format = 'csr', dtype = np.float64);
	# A += sp.csr_matrix(([-1.25, 0.25, 0.25, 0.25], ([0, 0, 1, N - 1], [N - 1, N - 2, N - 1, 0])), shape = (N, N));
	return A;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_LUST(N):
	A_GL = Adv_mat_Gauss_linear(N);
	A_LU = Adv_mat_Gauss_linearUpwind(N);
	A = 0.25*A_LU + 0.75*A_GL;
	return A;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Burgers_Gauss_linear(N):
	A = Adv_mat_Gauss_linear(N);
	phi = sp.diags(0.25*np.ones(N - 1, dtype = np.float64), offsets = 1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(0.25*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(0.5*np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64);
	phi += sp.csr_matrix(([0.25, 0.25], ([0, N - 1], [N - 1, 0])), shape = (N, N));
	return A, phi;

def Adv_mat_Burgers_Gauss_upwind(N):
	A = Adv_mat_Gauss_upwind(N);
	phi = sp.diags(0.5*np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(0.5*np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64);
	phi += sp.csr_matrix(([0.5], ([0], [N - 1])), shape = (N, N));
	return A, phi;

def Adv_mat_Burgers_Gauss_linearUpwind(N):
	A = Adv_mat_Gauss_linearUpwind(N);
	phi = sp.diags(np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64);
	return A, phi;

@functools.lru_cache(maxsize = None, typed = False)
def Adv_mat_Burgers_LUST(N):
	A_GL, phi_GL = Adv_mat_Burgers_Gauss_linear(N);
	A_LU, phi_LU = Adv_mat_Burgers_Gauss_linearUpwind(N);
	A = 0.25*A_LU + 0.75*A_GL;
	phi = 0.25*phi_LU + 0.75*phi_GL;
	return A, phi;

@functools.lru_cache(maxsize = None, typed = False)
def Diff_mat_Gauss_linear(N):
	D = sp.diags(np.ones(N - 1, dtype = np.float64), offsets = 1, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(-2.0*np.ones(N, dtype = np.float64), offsets = 0, shape = (N, N), format = 'csr', dtype = np.float64) \
	  + sp.diags(np.ones(N - 1, dtype = np.float64), offsets = -1, shape = (N, N), format = 'csr', dtype = np.float64);
	D += sp.csr_matrix(([1, 1], ([0, N - 1], [N - 1, 0])), shape = (N, N));
	return D;

def Compute_eigVals(divScheme, laplacianScheme, N, alpha, kdt, sigma_f):
	A = divScheme(N);
	D = laplacianScheme(N);
	ADt = -A*alpha + D*kdt;
	beta = 2*np.pi*np.arange(0, N, 1)/N;
	beta[beta > np.pi] = 2*np.pi - beta[beta > np.pi];
	JM = np.arange(N)*np.arange(N)[:, None];
	ldt = np.sum(ADt[0, :].toarray()*np.exp(1j*2.0*np.pi*JM/N), axis = -1);
	sigma = sigma_f(ldt);
	magSig = np.abs(sigma);
	relPse = np.ones(N - 1);
	relPse = -np.angle(sigma[1:])/(alpha*beta[1:]);
	relPse[relPse < 0] *= -1;
	return ldt, beta, magSig, relPse, ADt;

def plot_stable_region(lam_sig_relation, ldt, save_plots, save_name, ax = None):
	x = np.linspace(-4, 4, 100);
	X = np.meshgrid(x, x);
	z = X[0] + 1j*X[1];
	Rlevel = abs(lam_sig_relation(z));
	if ax == None:
		fig = plt.figure(figsize = (5, 5));
		ax = fig.add_subplot(111);
	ax.contourf(x, x, Rlevel, [1, 1000], hatches = ['//'], cmap = 'gray', alpha = 0.9);
	ax.contour(x, x, Rlevel, [1, 1000]);
	ax.set_xlabel(r'$Re(\lambda \Delta t)$', fontsize = 13);
	ax.set_ylabel(r'$Im(\lambda \Delta t)$', fontsize = 13);
	ax.plot([0, 0], [-4, 4], '-k');
	ax.plot([-4, 4], [0, 0], '-k');
	Re = ldt.real;
	Im = ldt.imag;
	ax.scatter(Re, Im, color = 'red', marker = 'x', label = r'$\lambda \Delta t$', s = 4);
	ax.legend(fontsize = 12);
	if (save_plots): plt.savefig('%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def plot_stable_region2(lam_sig_relation, ldt_arr, labels, save_plots, save_name, ax = None):
	x = np.linspace(-4, 4, 100);
	X = np.meshgrid(x, x);
	z = X[0] + 1j*X[1];
	Rlevel = abs(lam_sig_relation(z));
	if ax == None:
		fig = plt.figure(figsize = (5, 5));
		ax = fig.add_subplot(111);
	ax.set_prop_cycle(color = ['b', 'orange', 'r'],
						marker = ['o', '+', 'x']);
	ax.contourf(x, x, Rlevel, [1, 1000], hatches = ['//'], cmap = 'gray', alpha = 0.9);
	ax.contour(x, x, Rlevel, [1, 1000]);
	ax.set_xlabel(r'$Re(\lambda \Delta t)$', fontsize = 13);
	ax.set_ylabel(r'$Im(\lambda \Delta t)$', fontsize = 13);
	ax.plot([0, 0], [-4, 4], '-k');
	ax.plot([-4, 4], [0, 0], '-k');
	for i, ldt in enumerate(ldt_arr):	
		Re = ldt.real;
		Im = ldt.imag;
		ax.scatter(Re, Im, label = r'$\lambda \Delta t$ (%s)'%labels[i], s = 4);
	ax.legend(fontsize = 12);
	if (save_plots): plt.savefig('%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def testConvergence(errFunc, N_range, alpha_range, args, save_plots, save_name):
	err = np.zeros((N_range.shape[0], alpha_range.shape[0]), dtype = np.float64);
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_prop_cycle(color = ['b', 'orange', 'r', 'k', 'm'],
						marker = ['o', '+', 'x', '*', 's']);
	for i, N in enumerate(N_range):
		for j, alpha in enumerate(alpha_range):
			err[i, j] = errFunc(N, alpha, *args);
		ax.loglog(alpha_range, err[i, :], label = 'N = %i'%(N), linewidth = 0.75, markersize = 3.5);
	# annotation.slope_marker((1.2, 0.05), expOrder, ax = ax);
	ax.legend(loc = 'best', fontsize = 12);
	ax.set_xlabel(r'$\alpha$ [-]', fontsize = 15);
	ax.set_ylabel(r'$\varepsilon$ [-]', fontsize = 15);
	ax.tick_params(axis = 'x', labelsize = 13);
	ax.tick_params(axis = 'y', labelsize = 13);
	ax.minorticks_on();
	ax.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	if (save_plots): plt.savefig('./FiguresPDF/%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def ordOfAccTest(N_range, scheme, c, nu):
	u_ext = lambda x: np.sin(2*np.pi*x);
	f = lambda x: 2*np.pi*c*np.cos(2*np.pi*x) + 4*np.pi**2*nu*np.sin(2*np.pi*x);
	rms_e = np.zeros(N_range.shape[0]);
	for i, N in enumerate(N_range):
		x = np.linspace(0, 1, N)[1:-1];
		dx = x[1] - x[0];
		A = c*scheme(N)[1:-1, 1:-1] - nu/dx*Diff_mat_Gauss_linear(N)[1:-1, 1:-1];
		rms_e[i] = 1/np.sqrt(N)*np.linalg.norm(la.spsolve(A, f(x)*dx) - u_ext(x));
	return rms_e;

def plotOrdOfAcc(save_plots):
	c, nu = 1., 0.1;
	save_name = 'ordOfAccTest'
	N_range = np.logspace(1, 3, 10, base = 10, dtype = int);
	divSchemes = [Adv_mat_Gauss_linear, Adv_mat_Gauss_linearUpwind, Adv_mat_Gauss_upwind];
	divSchemesName = ['Gauss linear', 'Gauss linearUpwind', 'Gauss upwind'];
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_prop_cycle(color = ['b', 'orange', 'r', 'k', 'm'],
						marker = ['o', '+', 'x', '*', 's']);
	for i, scheme in enumerate(divSchemes):
		rms_e = ordOfAccTest(N_range, scheme, c, nu);
		ax.loglog(N_range, rms_e, linewidth = 0.85, markersize = 3.5, label = divSchemesName[i]);
	slope_marker((100, 0.0003), (-2, 1), invert = True);
	slope_marker((210, 0.01), (-1, 1), invert = True);
	ax.legend(loc = 'best');
	ax.set_xlabel(r'$N$ [-]', fontsize = 13);
	ax.set_ylabel(r'$\varepsilon$ [-]', fontsize = 13);
	ax.minorticks_on();
	ax.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	if (save_plots): plt.savefig('./FiguresPDF/%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	plt.show();
	return 0;

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.rcParams['mathtext.fontset'] = 'stix'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
	from mpltools.annotation import slope_marker
	# N = 10;
	# A = Adv_mat_Gauss_linear(N);
	# D = Diff_mat_Gauss_linear(N);

	# fig = plt.figure();
	# ax = fig.add_subplot(111);
	# ax.imshow(A.toarray());
	# fig = plt.figure();
	# ax = fig.add_subplot(111);
	# ax.imshow(D.toarray());
	plotOrdOfAcc(True);
	plt.show();
