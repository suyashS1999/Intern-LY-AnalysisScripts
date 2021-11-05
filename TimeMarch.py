import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as pltani

def TimeMarch(u0, ADt, dt, Ndt, ddtScheme, S = None, nonLin = False, updateFunc = None, nCorrector = 1, x = None, dx = None, alpha = None, kdt = None):
	u = np.zeros((u0.shape[0], Ndt), dtype = np.float64);
	u[:, 0] = u0;
	time = np.zeros(Ndt, dtype = np.float64);
	if S == None: S = lambda x, t: np.zeros((u0.shape[0], t.shape[0]));
	if kdt != None: nu = kdt*dx**2/dt;
	for t in range(1, Ndt):
		time[t] = time[t - 1] + dt;
		SDt = S(x, time[0:t + 1])*dt;
		if not nonLin:
			u[:, t] = ddtScheme(u[:, 0:t], ADt, SDt[:, 0:t + 1]);
		else:
			c = u[:, t - 1];
			for n in range(nCorrector):
				ADt = updateFunc(c, alpha/max(np.absolute(u[:, t - 1])), kdt);
				u[:, t] = ddtScheme(u[:, 0:t], ADt, SDt[:, 0:t + 1]);
				dt = alpha*dx/max(np.absolute(u[:, t]));
				kdt = nu*dt/dx**2;
				c = u[:, t];
	return u, time;

def BDF1(u, ADt, SDt):
	uNew = la.spsolve(sp.eye(*(ADt.shape), format = 'csr') - ADt, u[:, -1] + SDt[:, -1]);
	return uNew;

def BDF2(u, ADt, SDt):
	if u.shape[1] == 1: return BDF1(u, ADt, SDt);
	else:
		uNew = la.spsolve(3/2*sp.eye(*(ADt.shape), format = 'csr') - ADt, (2*u[:, -1] - 1/2*u[:, -2]) + SDt[:, -1]);
		return uNew;

def FDF1(u, ADt, SDt):
	uNew = (sp.eye(*(ADt.shape), format = 'csr') + ADt)@(u[:, -1] +  SDt[:, -2]);
	return uNew;

def Crank_Nicolson(u, ADt, SDt):
	uNew = la.spsolve(sp.eye(*(ADt.shape), format = 'csr') - 0.5*ADt, u[:, -1] + 0.5*ADt@u[:, -1] + 0.5*(SDt[:, -1] + SDt[:, -2]));
	return uNew;

def sigma_BDF1(ldt): return 1/(1 - ldt);
def sigma_BDF2(ldt): return (2 + np.sqrt(4 - 2*(3/2 - ldt)))/(2*(3/2 - ldt));		#(4/3 + np.sqrt(16/9 - 4/3*(1 - 2/3*ldt)))/(2*(1 - 2/3*ldt));
def sigma_FDF1(ldt): return 1 + ldt;
def sigma_Crank_Nicolson(ldt): return (1 + 0.5*ldt)/(1 - 0.5*ldt);

def Animate(x, u, time, xExact = None, uExact = None, ylims = [0.0, 1.0]):
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_ylim(ylims);
	ax.set_xlabel(r'$x$', fontsize = 13);
	ax.set_ylabel(r'$u$', fontsize = 13);
	line1 = ax.plot(x, x, '-', label = 'Numerical Solution', markersize = 3)[0];
	line2 = ax.plot(x, x, '--', alpha = 0.85, label = 'Exact Solution')[0];
	check_bool = isinstance(uExact, np.ndarray);
	if (check_bool) == False: line2.set_visible(False); 
	else: line2.set_data(xExact, xExact);
	text = ax.text(0.05, 0.9, '', transform = ax.transAxes);
	if (check_bool): ax.legend(loc = 'best');

	def init():
		line1.set_ydata(np.ma.array(x, mask = True));
		if (check_bool): line2.set_ydata(np.ma.array(xExact, mask = True));
		text.set_text('');
		return line1, line2, text;

	def step(i):
		line1.set_ydata(u[:, i]);
		if (check_bool): line2.set_ydata(uExact[:, i]);
		text.set_text('time = %.3fs'%(time[i]));
		return line1, line2, text;
	
	ani = pltani.FuncAnimation(fig, step, u.shape[1], init_func = init, interval = 25, blit = True);
	plt.show();

def ordOfAccTest(Ndt_range, scheme, tmax, m, k):
	A = sp.csr_matrix(([-k/m, 1], ([0, 1], [1, 0])), shape = (2, 2));
	u_ext = lambda t: np.cos(np.sqrt(k/m)*t);
	u0 = np.array([0.0, 1.0]);
	rms_e = np.zeros(Ndt_range.shape[0]);
	for i, Ndt in enumerate(Ndt_range):
		dt = tmax/(Ndt - 1);
		U, t = TimeMarch(u0, A*dt, dt, Ndt, scheme);
		rms_e[i] = 1/(Ndt)*np.linalg.norm(U[1, :] - u_ext(t));
	return rms_e;

def plotOrdOfAcc(save_plots):
	tmax = 5;
	m, k = 1.0, 4*np.pi**2;
	save_name = 'ordOfAccTest_ddt'
	Ndt_range = np.logspace(2, np.log10(5000), 10, base = 10, dtype = int);
	ddtSchemes = [BDF1, BDF2, Crank_Nicolson];
	ddtSchemesName = ['BDF1', 'BDF2', 'Crank Nicolson'];
	fig = plt.figure();
	ax = fig.add_subplot(111);
	ax.set_prop_cycle(color = ['b', 'orange', 'r', 'k', 'm'],
						marker = ['o', '+', 'x', '*', 's']);
	for i, scheme in enumerate(ddtSchemes):
		rms_e = ordOfAccTest(Ndt_range, scheme, tmax, m, k);
		ax.loglog(Ndt_range, rms_e, linewidth = 0.85, markersize = 3.5, label = ddtSchemesName[i]);
	slope_marker((710, 5.5e-5), (-2, 1), invert = True);
	slope_marker((1900, 0.002), (-1, 1), invert = True);
	ax.legend(loc = 'best');
	ax.set_xlabel(r'$N_{dt}$ [-]', fontsize = 13);
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

	plotOrdOfAcc(False);