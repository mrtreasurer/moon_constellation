import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import minimize

import constants as cte

from optim_class import Coverage, initiate


sim_time, sun, targets = initiate()
coverage = Coverage(sim_time, sun, targets)

res = minimize(coverage.fitness, np.array([cte.h_crit, np.radians(50), 2*np.pi/4, 2*np.pi/3]), method="Nelder-Mead")
print(res.x, res.fun)

sats = coverage.create_constellation(res.x[0], res.x[1], int(2*np.pi//res.x[2]), int(2*np.pi//res.x[3]))
charge = coverage.propagate_constellation(sun, targets, sats, cte.r_m, res.x[0], cte.min_elev, cte.max_sat_range, cte.tar_battery_cap, cte.tar_charge_power - cte.tar_op_power, cte.dt, cte.sat_las_power, cte.tar_hib_power)[0]

sat_period = 2*np.pi * np.sqrt(res.x[0]**3/cte.mu_m)

plt.figure(1)
ax = plt.axes(projection='3d')

for i in range(sats.shape[1]):
    ax.plot(sats[:, i, 0][sim_time < sat_period], sats[:, i, 1][sim_time < sat_period], sats[:, i, 2][sim_time < sat_period], 'b')
    ax.plot([sats[0, i, 0]], [sats[0, i, 1]], [sats[0, i, 2]], 'bo')
    
for i in range(targets.shape[1]):
    ax.plot(targets[::100, i, 0], targets[::100, i, 1], targets[::100, i, 2], 'g')
    ax.plot([targets[0, i, 0]], [targets[0, i, 1]], [targets[0, i, 2]], 'go')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

#plt.figure(2)
#for i, tar in enumerate(targets):
#    plt.subplot(2, 3, i + 1)
#    plt.plot(sim_time, tar.n_sats_in_range)
#
plt.figure(3)
for i in range(targets.shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.plot(sim_time, charge[:, i])

#plt.figure(4)
#for i in range(targets.shape[1]):
#    plt.subplot(2, 4, i + 1)
#    plt.plot(sim_time[contact[:, i, 0]], eff[:, i, 0][contact[:, i, 0]])

#plt.figure(5)
#for i in range(cos_elev.shape[1]):
#    plt.subplot(2, 4, i + 1)
#    plt.plot(sim_time, cos_elev[:, i])

plt.show()