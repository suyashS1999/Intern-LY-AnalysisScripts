import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# from libs.linearAdvec_mat import Compute_eigVals as Compute_eigVals_cy
# from linearAdvec_mat_py import Compute_eigVals as Compute_eigVals_py
import FVM
import TimeMarch as ddt
from scipy import signal
import os
import re
#%% ------------------ Inputs ---------------------
N = 320;								# Number of mesh points
alpha = -1.0;							# CFL number c*dt/dx
c = -1.0;								# Convection speed
nu = 0.00;								# Diffusion constant
tmax = 0.5;								# Max time
NTest = 10*np.logspace(2, 4, 3, base = 2, dtype = int);
alphaTest = np.logspace(-1, np.log10(5), 10, base = 10);
save_plots = False;
fvSchemes = {
	'divScheme': FVM.Adv_mat_Gauss_linearUpwind, 
	'laplacianScheme': FVM.Diff_mat_Gauss_linear, 
	'ddtScheme': ddt.BDF2
};
#%% --------------- Computed values ----------------
xExact = np.linspace(0, 1, 500);		# mesh points for exact solution
sigma = getattr(ddt, 'sigma_%s'%(fvSchemes['ddtScheme'].__name__));
divSchemeName = re.search('Adv_mat_(.*)', fvSchemes['divScheme'].__name__).group(1);
ddtSchemeName = fvSchemes['ddtScheme'].__name__;
dirName = './FiguresPDF/%s'%(divSchemeName);
if os.path.isdir(dirName) == False: os.makedirs(dirName);
SimName = 'N_%i_alpha_%0.1f_nu_%0.2e_divScheme_%s_ddtScheme_%s'%(N, alpha, nu, divSchemeName, ddtSchemeName);
u0 = lambda x: 0.5*(signal.square(2*np.pi*(x - 0.1), duty = 0.2) + 1);	# Initial condition
uExact_func = lambda x, t: 0.5*(signal.square(2*np.pi*(x[:, np.newaxis] - 0.1 - c*t), duty = 0.2) + 1);
#%% ---------------- Run Analysis ------------------
def runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes):
	x = np.linspace(0, 1, N);				# mesh points
	dx = 1/(N - 1);							# dx
	dt = alpha*dx/c;						# Time step
	Ndt = int(tmax//dt);					# Number of time steps
	kdt = nu*dt/dx**2;						# nu*dt/dx^2
	ldt, beta, magSig, relPse, ADt = FVM.Compute_eigVals(fvSchemes['divScheme'], fvSchemes['laplacianScheme'], N, alpha, kdt, sigma);
	u, time = ddt.TimeMarch(u0(x), ADt, dt, Ndt, fvSchemes['ddtScheme']);
	return ldt, beta, magSig, relPse, x, u, time;

def solnVerif(N, alpha, nu, sigma, u0, tmax, fvSchemes, uExact_func):
	_, _, _, _, x, u, t = runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes);
	uExact = uExact_func(x, t);
	err = np.sqrt(np.mean(((uExact - u)**2).flatten()));
	return err;

def Plot_amp_phase_err(beta, magSig, relPse, save_plots, save_name):
	fig = plt.figure();
	ax1 = fig.add_subplot(111);
	ax1.set_xlabel(r'$\beta$ [-]', fontsize = 13);
	ax1.set_ylabel(r'$|\sigma|$ [-]', fontsize = 13);
	ax1.tick_params(axis = 'x', labelsize = 13);
	ax1.tick_params(axis = 'y', labelsize = 13);
	ax1.set_xlim((0, np.pi));
	ax1.grid(True);
	ax1.set_prop_cycle(color = ['b', 'orange', 'r', 'k', 'm'],
						marker = ['o', '+', 'x', '*', 's']);
	plt.minorticks_on();
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax1.plot(beta, magSig, linewidth = 0.75, markersize = 3);
	# ax1.legend(loc = 'best', fontsize = 12);
	if (save_plots): plt.savefig('%s_amp.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);

	fig = plt.figure();
	ax2 = fig.add_subplot(111);
	ax2.set_xlabel(r'$\beta$ [-]', fontsize = 13);
	ax2.set_ylabel(r'$\angle \rho_{rel}$ [-]', fontsize = 13);
	ax2.tick_params(axis = 'x', labelsize = 13);
	ax2.tick_params(axis = 'y', labelsize = 13);
	ax2.set_xlim((0, np.pi));
	ax2.grid(True);
	ax2.set_prop_cycle(color = ['b', 'orange', 'r', 'k', 'm'],
						marker = ['o', '+', 'x', '*', 's']);
	plt.minorticks_on();
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax2.plot(beta[1:], relPse, linewidth = 0.75, markersize = 3);
	# ax2.legend(loc = 'best', fontsize = 12);
	if (save_plots): plt.savefig('%s_pse.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def Plot_solution(x, u, t_idx, save_plots, save_name, uExact = None):
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_xlim([0.0, 1.0]);
	ax.set_ylim([-0.01, 1.5]);
	ax.set_xlabel(r'$x$', fontsize = 15);
	ax.set_ylabel(r'$u$', fontsize = 15);
	ax.tick_params(axis = 'x', labelsize = 13);
	ax.tick_params(axis = 'y', labelsize = 13);
	ax.grid(True);
	plt.minorticks_on();
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax.plot(x, u[:, t_idx], '-', label = 'Numerical Solution', linewidth = 0.9, markersize = 2.5);
	if isinstance(uExact, np.ndarray): ax.plot(xExact, uExact[:, t_idx], '--', alpha = 0.85, label = 'Exact Solution', linewidth = 1); ax.legend(loc = 'best', fontsize = 12);
	if (save_plots): plt.savefig('%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def main():
	ldt, beta, magSig, relPse, x, u, time = runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes);
	uExact = uExact_func(xExact, time);
	FVM.plot_stable_region(sigma, ldt, save_plots, '%s/stab_reg_%s'%(dirName, SimName));
	# Plot the amplification factor and relative phase
	Plot_amp_phase_err(beta, magSig, relPse, save_plots, '%s/amp_pse_err_%s'%(dirName, SimName));
	# Plot solution at intial and final time level
	Plot_solution(x, u, -1, save_plots, '%s/u_final_%s'%(dirName, SimName), uExact);
	Plot_solution(x, u, 0, save_plots, '%s/u_initial_condition_N_%i'%(dirName, N), uExact);
	# FVM.testConvergence(solnVerif, NTest, alphaTest, (nu, sigma, u0, tmax, fvSchemes, uExact_func), save_plots, '');
	plt.show();
	ddt.Animate(x, u, time, xExact, uExact, ylims = [-0.01, 1.5]);
	return 0;

if __name__ == '__main__':
	main();




# st = time.time();
# eigs = Compute_eigVals_cy(N, alpha, kdt);
# t_cy = time.time() - st;
# print('Cython time : %0.8fs\n'%(t_cy));

# st = time.time();
# eigs = Compute_eigVals_py(N, alpha, kdt);
# t_py = time.time() - st;
# print('Python time : %0.8fs\n'%(t_py));

# st = time.time();
# eigs = Compute_eigVals_vec(N, alpha, kdt);
# t_py_vec = time.time() - st;
# print('vect-Python time : %0.8fs\n'%(t_py_vec));
# print('Cython is %0.5f times faster'%(t_py/t_cy));

