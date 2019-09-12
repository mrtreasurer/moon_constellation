# -*- coding: utf-8 -*-

import numpy as np

import sps
import targets as tgt

from periodic_orbits.constants import r_sh

import constants as cte
        
sin_rho = r_sh/sps.sma
cos_min_elev = np.cos(np.radians(10))

contact = np.zeros((len(cte.sim_time), len(tgt.targets), len(sps.sats)), dtype=bool)
dist = np.zeros((len(cte.sim_time), len(tgt.targets), len(sps.sats)))
eff = np.zeros((len(cte.sim_time), len(tgt.targets), len(sps.sats)))

for i in range(len(cte.sim_time)):
    
    for j, tar in enumerate(tgt.targets):
        tpos = tar.pos[i]
        
        for k, sat in enumerate(sps.sats):
            spos = sat.pos[i]
            
            st = spos - tpos
            
            if np.linalg.norm(st) <= sps.rge:
                sin_n = np.sqrt(1 - (np.dot(st, spos)/(np.linalg.norm(st)*np.linalg.norm(spos)))**2)
            
                cos_elev = sin_n/sin_rho
                
                if cos_elev < cos_min_elev:
                    contact[i, j, k] = True
                    dist[i, j, k] = np.linalg.norm(st)
                    
                    tar.in_range(i, sat)
                    sat.in_range(i, tar)
    
    if i != 0:
        for j, tar in enumerate(tgt.targets):
            
            sps_power = 0            
            if i in tar.sats_in_range:
                
                for sat in tar.sats_in_range[i]:
                    sps_power += sat.link_eff(tar, i) * sat.laser_power/sat.n_targets_in_range[i]
                    eff[i, j, sat.ident] = sat.link_eff(tar, i)
                    
                tar.update_charge(i, sps_power)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.figure(1)
ax = plt.axes(projection='3d')

#for i in range(sps.n_sats):
#    ax.plot(sat_hist[i, :, 0], sat_hist[i, :, 1], sat_hist[i, :, 2])
#    ax.plot([sat_hist[i, 0, 0]], [sat_hist[i, 0, 1]], [sat_hist[i, 0, 2]], 'o')

for sat in sps.sats:
    ax.plot(sat.pos[:, 0][sat.time<sat.period], sat.pos[:, 1][sat.time<sat.period], sat.pos[:, 2][sat.time<sat.period], 'b')
    ax.plot([sat.pos[0, 0]], [sat.pos[0, 1]], [sat.pos[0, 2]], 'bo')
    
for tar in tgt.targets:
    ax.plot(tar.pos[::10, 0], tar.pos[::10, 1], tar.pos[::10, 2], 'g')
    ax.plot([tar.pos[0, 0]], [tar.pos[0, 1]], [tar.pos[0, 2]], 'go')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.figure(2)
for i, tar in enumerate(tgt.targets):
    plt.subplot(2, 3, i + 1)
    plt.plot(cte.sim_time, tar.n_sats_in_range)

plt.figure(3)
for i, tar in enumerate(tgt.targets):
    plt.subplot(2, 3, i + 1)
    plt.plot(cte.sim_time[tar.sunlight==False], tar.charge[tar.sunlight==False])
    
#plt.figure(4)
#for i, tar in enumerate(tgt.targets):
#    plt.subplot(2, 3, i + 1)
#    plt.plot(cte.sim_time, eff[:, i, :])
