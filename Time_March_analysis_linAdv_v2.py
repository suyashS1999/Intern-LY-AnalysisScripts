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
N = 160;								# Number of mesh points
alpha = 1.0;							# CFL number c*dt/dx
c = 1.0;								# Convection speed
nu = 0.00;								# Diffusion constant
tmax = 0.5;								# Max time
NTest = 10*np.logspace(2, 4, 3, base = 2, dtype = int);
alphaTest = np.logspace(-1, np.log10(5), 10, base = 10);
save_plots = False;
fvSchemes = {
	'divScheme': [FVM.Adv_mat_Gauss_linearUpwind, FVM.Adv_mat_Gauss_linear, FVM.Adv_mat_LUST], 
	'laplacianScheme': FVM.Diff_mat_Gauss_linear, 
	'ddtScheme': ddt.BDF2
};
#%% --------------- Computed values ----------------
xExact = np.linspace(0, 1, 500);		# mesh points for exact solution
sigma = getattr(ddt, 'sigma_%s'%(fvSchemes['ddtScheme'].__name__));
ddtSchemeName = fvSchemes['ddtScheme'].__name__;
dirName = './FiguresPDF/%s'%(ddtSchemeName);
if os.path.isdir(dirName) == False: os.makedirs(dirName);
SimName = 'N_%i_alpha_%0.1f_nu_%0.2e_ddtScheme_%s'%(N, alpha, nu, ddtSchemeName);
u0 = lambda x: 0.5*(signal.square(2*np.pi*(x - 0.1), duty = 0.2) + 1);	# Initial condition
uExact_func = lambda x, t: 0.5*(signal.square(2*np.pi*(x[:, np.newaxis] - 0.1 - c*t), duty = 0.2) + 1);
#%% ---------------- Run Analysis ------------------
def runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes, scheme_idx):
	x = np.linspace(0, 1, N);				# mesh points
	dx = 1/(N - 1);							# dx
	dt = alpha*dx/c;						# Time step
	Ndt = int(tmax//dt);					# Number of time steps
	kdt = nu*dt/dx**2;						# nu*dt/dx^2
	ldt, beta, magSig, relPse, ADt = FVM.Compute_eigVals(fvSchemes['divScheme'][scheme_idx], fvSchemes['laplacianScheme'], N, alpha, kdt, sigma);
	u, time = ddt.TimeMarch(u0(x), ADt, dt, Ndt, fvSchemes['ddtScheme']);
	return ldt, beta, magSig, relPse, x, u, time;

def solnVerif(N, alpha, nu, sigma, u0, tmax, fvSchemes, uExact_func):
	_, _, _, _, x, u, t = runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes);
	uExact = uExact_func(x, t);
	err = np.sqrt(np.mean(((uExact - u)**2).flatten()));
	return err;

def Plot_amp_phase_err(beta, magSig_arr, relPse_arr, labels, save_plots, save_name):
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
	for i, magSig in enumerate(magSig_arr): ax1.plot(beta, magSig, label = labels[i], linewidth = 0.75, markersize = 3);
	ax1.legend(loc = 'best', fontsize = 12);
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
	for i, relPse in enumerate(relPse_arr): ax2.plot(beta[1:], relPse, label = labels[i], linewidth = 0.75, markersize = 3);
	ax2.legend(loc = 'best', fontsize = 12);
	if (save_plots): plt.savefig('%s_pse.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def Plot_solution(x, u_arr, t_idx, labels, save_plots, save_name, uExact = None):
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_prop_cycle(color = ['b', 'orange', 'r']);
	ax.set_xlim([0.0, 1.0]);
	ax.set_ylim([-0.01, 1.75]);
	ax.set_xlabel(r'$x$', fontsize = 15);
	ax.set_ylabel(r'$u$', fontsize = 15);
	ax.tick_params(axis = 'x', labelsize = 13);
	ax.tick_params(axis = 'y', labelsize = 13);
	ax.grid(True);
	plt.minorticks_on();
	plt.grid(b = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	for i, u in enumerate(u_arr): ax.plot(x, u[:, t_idx], '-', label = 'Numerical Solution with %s'%(labels[i]), linewidth = 0.9, markersize = 2.5);
	if isinstance(uExact, np.ndarray): ax.plot(xExact, uExact[:, t_idx], '--k', alpha = 0.85, label = 'Exact Solution', linewidth = 1); ax.legend(loc = 'best', fontsize = 12);
	if (save_plots): plt.savefig('%s.pdf'%(save_name), dpi = 300, bbox_inches = 'tight');
	else: plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0);
	return 0;

def main():
	ldt_arr = np.zeros((len(fvSchemes['divScheme']), N), dtype = complex);
	magSig_arr = np.zeros((len(fvSchemes['divScheme']), N), dtype = np.float64);
	relPse_arr = np.zeros((len(fvSchemes['divScheme']), N - 1), dtype = np.float64);
	u_arr = np.zeros((len(fvSchemes['divScheme']), N, int(tmax//(alpha/(c*(N - 1))))), dtype = np.float64);
	labels = np.zeros(len(fvSchemes['divScheme']), dtype = object)
	for idx in range(len(fvSchemes['divScheme'])):
		ldt_arr[idx, :], beta, magSig_arr[idx, :], relPse_arr[idx, :], x, u_arr[idx, ...], time = runSim(N, alpha, nu, sigma, u0, tmax, fvSchemes, idx);
		labels[idx] = re.search('Adv_mat_(.*)', fvSchemes['divScheme'][idx].__name__).group(1).replace('_', ' ');
	uExact = uExact_func(xExact, time);
	
	FVM.plot_stable_region2(sigma, ldt_arr, labels, save_plots, '%s/stab_reg_%s'%(dirName, SimName));
	# Plot the amplification factor and relative phase
	Plot_amp_phase_err(beta, magSig_arr, relPse_arr, labels, save_plots, '%s/amp_pse_err_%s'%(dirName, SimName));
	# Plot solution at intial and final time level
	Plot_solution(x, u_arr, -1, labels, save_plots, '%s/u_final_%s'%(dirName, SimName), uExact);
	Plot_solution(x, u_arr, 0, labels, save_plots, '%s/u_initial_condition_N_%i'%(dirName, N), uExact);
	# FVM.testConvergence(solnVerif, NTest, alphaTest, (nu, sigma, u0, tmax, fvSchemes, uExact_func), save_plots, '');
	plt.show();
	# ddt.Animate(x, u_arr[0, :], time, xExact, uExact, ylims = [-0.01, 1.5]);
	return 0;

if __name__ == '__main__':
	main();

