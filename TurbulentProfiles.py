import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import glob

files = glob.glob('./TurbulentInflowData/*');

points = 15*np.genfromtxt(files[1]);
points[:, [1, 2]] = points[:, [2, 1]];
R = np.genfromtxt(files[0]);
L = np.genfromtxt(files[-1]);

fig = plt.figure(figsize = (10, 5));
ax1 = fig.add_subplot(121);
ax1.plot(points[:, -1], R[:, 0], label = r"$<u'u'>$");
ax1.plot(points[:, -1], R[:, 1], label = r"$<u'v'>$");
ax1.plot(points[:, -1], R[:, 3], label = r"$<v'v'>$");
ax1.plot(points[:, -1], R[:, 5], label = r"$<w'w'>$");
ax1.grid(True);
ax1.legend();

ax2 = fig.add_subplot(122);
ax2.plot(points[:, -1], L);
ax2.grid(True);
plt.show();