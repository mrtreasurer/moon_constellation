# -*- coding: utf-8 -*-

import numpy as np
import pygmo as pg

from matplotlib import pyplot as plt

import constants as cte

from optim_class import Coverage, initiate

sim_time, sun_pos, targets = initiate()

# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
coverage = Coverage(sim_time, sun_pos, targets)

prob = pg.problem(coverage)

# 2 - Instantiate a pagmo algorithm
algo = pg.algorithm(pg.gaco(gen=100))

# 3 - Instantiate an archipelago with 16 islands having each 20 individuals
archi = pg.archipelago(1, algo=algo, prob=prob, pop_size=7)

# 4 - Run the evolution in parallel on the 16 separate islands 10 times.
archi.evolve(1)  

# 5 - Wait for the evolutions to be finished
archi.wait()

# 6 - Print the fitness of the best solution in each island
res = None
fit = 100
for isl in archi:
    if fit < isl.get_population().champion_f:
        fit = isl.get_population().champion_f
        res = isl.get_population().champion_x
 
sats = coverage.create_constellation(res[0], res[1], int(2*np.pi//res[2]), int(2*np.pi//res[3]))
charge = coverage.propagate_constellation(sun_pos, targets, sats, cte.r_m, res[0], cte.min_elev, cte.max_sat_range, cte.tar_battery_cap, cte.tar_charge_power - cte.tar_op_power, cte.dt, cte.sat_las_power, cte.tar_hib_power)[0]

sat_period = 2*np.pi * np.sqrt(res[0]**3/cte.mu_m)

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