# -*- coding: utf-8 -*-

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

import constants as cte
import definitions as d
import sps
import targets as tgt


def propagate_constellation(sun, targets, sats, r_m, sma, min_elev, max_range, bat_cap, charge_power, dt, sat_power, hib_power):
    cos_min_elev = np.cos(min_elev)
    sin_rho = r_m / sma
    sin_max_nadir = sin_rho * cos_min_elev
    cos_max_lambda = np.cos(np.pi/2 - min_elev - np.arcsin(sin_max_nadir))
    
    dist_vec = np.zeros((targets.shape[0], targets.shape[1], sats.shape[1], 3))
    
    for i in range(sats.shape[1]):
        dist_vec[:, :, i] = np.transpose(np.stack([np.subtract(sats[:, i], targets[:, j]) for j in range(targets.shape[1])]), (1, 0, 2))
        
    dist = np.linalg.norm(dist_vec, axis=3)
    
    einsum_targets = np.sqrt(np.einsum("ijm,ijm->ij", targets, targets))
    einsum_sats = np.sqrt(np.einsum("ijm,ijm->ij", sats, sats))
    
    cos_lambda = np.einsum("ijk,imk->imj", sats, targets) / \
    (np.concatenate([einsum_sats[:, None, :]]*targets.shape[1], axis=1) * \
     np.concatenate([einsum_targets[:, :, None]]*sats.shape[1], axis=2))
    
    contact = (cos_lambda > cos_max_lambda) * (dist < max_range)
    
    eff = d.link_eff(dist, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec)
    
    einsum_sun = np.sqrt(np.einsum("ij,ij->i", sun, sun))
    
    cos_sun_angle_t = np.einsum("ik,ijk->ij", sun, targets) / \
    (np.concatenate([einsum_sun[:, None]]*targets.shape[1], axis=1) * 
     einsum_targets)
    
    targets_in_sunlight = cos_sun_angle_t > 0
       
#    cos_sun_angle_s = np.einsum("ik,ijk->ij", sun, sats) / \
#    (np.concatenate([einsum_sun[:, None]]*sats.shape[1], axis=1) * 
#     einsum_sats)
#    
#    sats_in_sunlight = (cos_sun_angle_s > 0) + (np.sqrt(1 - cos_sun_angle_s**2) > r_m/sma)
    
    target_charge = bat_cap * np.ones((targets.shape[0:2]))
    
    for i in range(1, targets.shape[0]):
        for j in range(targets.shape[1]):
            
            if targets_in_sunlight[i, j]:
                target_charge[i, j] = min(bat_cap, target_charge[i-1, j] + charge_power * dt/3600)
                
            else:
                sps_power = 0                
                for k in range(sats.shape[1]):
                    if contact[i, j, k]:
                        n_targets = dict(zip(*np.unique(contact[i, :, k][np.logical_not(targets_in_sunlight[i])], return_counts=True)))[True]
                        
                        sps_power += eff[i, j, k] * sat_power / n_targets
                        
                target_charge[i, j] = min(500, max(0, target_charge[i-1, j] + (sps_power - hib_power) * dt/3600))
        
    return target_charge


def calculate_fitness(targets, n_sats):
    penalty = 1
    average_charges = []
    for tar in targets:
        average_charges.append(np.average(tar.charge))
        
        if not tar.alive:
            penalty = 100
            
    average_charge = np.average(average_charges)
            
    fitness = n_sats * 1/average_charge * penalty
    
    return fitness
                    

sim_time = np.arange(0, cte.moon_period + cte.dt, cte.dt)
#sim_time = np.arange(0, 2*np.pi*np.sqrt(cte.h_crit**3/cte.mu_m), cte.dt)

sun_pos = d.sun_loc(sim_time, cte.omega_earth, cte.sun_pos0)

targets = tgt.create_targets(cte.target_coors, sim_time, cte.omega_moon)

n_targets = len(cte.target_coors)

sma = cte.h_crit
inc = np.radians(50)

sat_period = 2*np.pi * np.sqrt(sma**3/cte.mu_m)

n_planes = 4
n_sats_plane = 3
n_sats = n_planes * n_sats_plane

raan = np.repeat(np.linspace(0, 2*np.pi, n_planes + 1)[:-1], n_sats_plane)
ta = np.tile(np.linspace(0, 2*np.pi, n_sats_plane + 1)[:-1], n_planes)

kep = np.zeros((n_sats, 6))
kep[:, 0] = sma
kep[:, 1] = cte.ecc
kep[:, 2] = inc
kep[:, 3] = raan
kep[:, 4] = cte.aop
kep[:, 5] = ta

sats = sps.create_sats(kep, cte.mu_m, sim_time)

charge = propagate_constellation(sun_pos, targets, sats, cte.r_m, sma, cte.min_elev, cte.max_sat_range, cte.tar_battery_cap, cte.tar_charge_power, cte.dt, cte.sat_las_power, cte.tar_hib_power)

plt.figure(1)
_targets = np.transpose(targets, (1, 0, 2))
ax = plt.axes(projection='3d')

#for i in range(n_sats):
#    ax.plot(sats[:, i, 0][sim_time<sat_period], sats[:, i, 1][sim_time<sat_period], sats[:, i, 2][sim_time<sat_period], 'b')
#    ax.plot([sats[0, i, 0]], [sats[0, i, 1]], [sats[0, i, 2]], 'bo')
#    break

for i in range(n_sats):
    ax.plot(sats[:, i, 0][sim_time < sat_period], sats[:, i, 1][sim_time < sat_period], sats[:, i, 2][sim_time < sat_period], 'b')
    ax.plot([sats[0, i, 0]], [sats[0, i, 1]], [sats[0, i, 2]], 'bo')
    
for i in range(n_targets):
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
    plt.subplot(2, 4, i + 1)
    plt.plot(sim_time, charge[:, i])
#    
#plt.figure(4)
#for i in range(targets.shape[1]):
#    plt.subplot(2, 4, i + 1)
#    plt.plot(sim_time[contact[:, i, 0]], eff[:, i, 0][contact[:, i, 0]])

#plt.figure(5)
#for i in range(cos_elev.shape[1]):
#    plt.subplot(2, 4, i + 1)
#    plt.plot(sim_time, cos_elev[:, i])
