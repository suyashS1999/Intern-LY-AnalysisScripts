import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import FVM
import TimeMarch as ddt
import matplotlib.animation as pltani
from mpltools import annotation
#%% ------------------ Inputs ---------------------
N = 100;									# Number of mesh points
alpha = 0.5;								# CFL number c*dt/dx
c = 1;									# Max Convection speed
nu = 1e-3;								# Diffusion constant
tmax = 0.5;								# Max simulation time
NTest = 10*np.logspace(2, 4, 3, base = 2, dtype = int);
alphaTest = np.logspace(-1, np.log10(5), 10, base = 10);
fvSchemes = {
	'divScheme': FVM.Adv_mat_Burgers_Gauss_linear, 
	'laplacianScheme': FVM.Diff_mat_Gauss_linear, 
	'ddtScheme': ddt.Crank_Nicolson
};
save_plots = False;
#%% ------------------ Functions -------------------
sigma = getattr(ddt, 'sigma_%s'%(fvSchemes['ddtScheme'].__name__));
# u0 = lambda x: c*np.exp(-150*(x - 0.25)**2);	# Initial condition
u0 = lambda x: c*np.sin(2*np.pi*x);	# Initial condition
S = lambda x, t: c*(nu*np.exp(-nu*t)*np.sin(2*np.pi*(t - x[:, np.newaxis])) - np.exp(-nu*t)*2*np.pi*np.cos(2*np.pi*(x[:, np.newaxis] - t)) + \
			  c*np.exp(-nu*t)*np.sin(2*np.pi*(x[:, np.newaxis] - t))*np.exp(-nu*t)*2*np.pi*np.cos(2*np.pi*(x[:, np.newaxis] - t)) + \
			  nu*np.exp(-nu*t)*4*np.pi**2*np.sin(2*np.pi*(x[:, np.newaxis] - t)));
manifacturedSoln = lambda x, t: c*np.exp(-nu*t)*np.sin(2*np.pi*(x[:, np.newaxis] - t));
#%% ---------------- Run Analysis ------------------
def runSim(N, alpha, nu, u0, S, tmax, fvSchemes):
	x = np.linspace(0, 1, N);				# mesh points
	dx = 1/(N - 1);							# dx
	dt = alpha*dx/c;						# Time step (dynamically changed throughout the simulation)
	kdt = nu*dt/dx**2;						# nu*dt/dx^2
	Ndt = int(tmax/dt);
	A, phi = fvSchemes['divScheme'](N);
	D = fvSchemes['laplacianScheme'](N);
	ADt_func = lambda u, alpha_c, kdt: -alpha_c*A.multiply((phi@u)[:, np.newaxis]) + kdt*D;
	# ldt = np.linalg.eigvals(-alpha*A.toarray() + kdt*D.toarray());
	u, time = ddt.TimeMarch(u0(x), 0, dt, Ndt, fvSchemes['ddtScheme'], S = S, nonLin = True, updateFunc = ADt_func, nCorrector = 3, x = x, dx = 1/(N - 1), alpha = alpha, kdt = kdt);
	ldt = np.linalg.eigvals(ADt_func(u[:, 0], alpha, kdt).toarray());
	return x, u, time, ldt;

def solnVerif(N, alpha, nu, u0, tmax, fvSchemes, manifacturedSoln, S):
	x, u, t, _ = runSim(N, alpha, nu, u0, S, tmax, fvSchemes);
	uExact = manifacturedSoln(x, t);
	err = np.sqrt(np.mean(((uExact - u)**2).flatten()));
	return err;

def main():
	# x, u, time, ldt = runSim(N, alpha, nu, u0, S, tmax, fvSchemes);
	# uExact = manifacturedSoln(x, time);
	# FVM.plot_stable_region(sigma, ldt, False, '');

	# Animate(x, u, time, uExact);
	# x, u, time, ldt = runSim(N, alpha, nu, u0, None, tmax, fvSchemes);
	# FVM.plot_stable_region(sigma, ldt, False, '');
	# ddt.Animate(x, u, time, ylims = [-1.5, 1.5]);
	FVM.testConvergence(solnVerif, NTest, alphaTest, (nu, u0, tmax, fvSchemes, manifacturedSoln, S), save_plots, 'Convergence_nu_%0.2e_divScheme_%s_ddtScheme_%s'%(nu, fvSchemes['divScheme'].__name__, fvSchemes['ddtScheme'].__name__));
	plt.show();
	return 0;

if __name__ == '__main__':
	main();