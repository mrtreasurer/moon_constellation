# -*- coding: utf-8 -*-

import numpy as np

import constants as cte
import definitions as d
import sps
import targets as tgt


def initiate():
    sim_time = np.arange(0, cte.moon_period + cte.dt, cte.dt)
    #sim_time = np.arange(0, 2*np.pi*np.sqrt(cte.h_crit**3/cte.mu_m), cte.dt)
    
    sun_pos = d.sun_loc(sim_time, cte.omega_earth, cte.sun_pos0)
    
    targets = tgt.create_targets(cte.target_coors, sim_time, cte.omega_moon)
    
    return sim_time, sun_pos, targets


class Coverage:
    def __init__(self, time, sun, targets):
        self.sim_time = np.arange(0, cte.moon_period + cte.dt, cte.dt)
        #sim_time = np.arange(0, 2*np.pi*np.sqrt(cte.h_crit**3/cte.mu_m), cte.dt)
        
        self.sun_pos = d.sun_loc(self.sim_time, cte.omega_earth, cte.sun_pos0)
        
        self.targets = tgt.create_targets(cte.target_coors, self.sim_time, cte.omega_moon)
                
    def fitness(self, x):
        sma = x[0]
        inc = x[1]
        plane_sep = x[2]
        sat_sep = x[3]
        
        n_planes = int(2*np.pi // plane_sep)
        n_sats_plane = int(2*np.pi // sat_sep)
                
        sats = self.create_constellation(sma, inc, n_planes, n_sats_plane)
        
        charge = self.propagate_constellation(self.sun_pos, self.targets, sats, cte.r_m, sma, cte.min_elev, cte.max_sat_range, cte.tar_battery_cap, cte.tar_charge_power, cte.dt, cte.sat_las_power, cte.tar_hib_power)
        
        fitness = (n_planes * n_sats_plane) * 1/float(np.mean(charge))
        
        return [fitness]
    
    def get_bounds(self):
        return ([cte.r_m, 0, 2*np.pi/8, 2*np.pi/8], [cte.max_sat_range, np.pi/2, np.pi, np.pi])

    def create_constellation(self, sma, inc, n_planes, n_sats_plane):        
        raan = np.repeat(np.linspace(0, 2*np.pi, n_planes + 1)[:-1], n_sats_plane)
        ta = np.tile(np.linspace(0, 2*np.pi, n_sats_plane + 1)[:-1], n_planes)
        
        kep = np.empty((n_planes * n_sats_plane, 6))
        kep[:, 0] = sma
        kep[:, 1] = cte.ecc
        kep[:, 2] = inc
        kep[:, 3] = raan
        kep[:, 4] = cte.aop
        kep[:, 5] = ta
        
        sats = sps.create_sats(kep, cte.mu_m, self.sim_time)
        
        return sats
    
    @staticmethod
    def propagate_constellation(sun, targets, sats, r_m, sma, min_elev, max_range, bat_cap, charge_power, dt, sat_power, hib_power):
        cos_min_elev = np.cos(min_elev)
        sin_rho = r_m / sma
        sin_max_nadir = sin_rho * cos_min_elev
        cos_max_lambda = np.cos(np.pi/2 - min_elev - np.arcsin(sin_max_nadir))
        
        dist_vec = np.empty((targets.shape[0], targets.shape[1], sats.shape[1], 3))
        
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
    
        n_targets = np.sum(contact * np.logical_not(targets_in_sunlight)[:, :, None], axis=1)
        sps_power = np.sum((eff * sat_power / np.clip(np.transpose([n_targets]*eff.shape[1], (1, 0, 2)), 1, np.inf)) * contact, axis=2) * np.logical_not(targets_in_sunlight)
        
        target_charge = np.zeros((targets.shape[0:2]))
        target_charge[0] = bat_cap
        
        for i in range(1, targets.shape[0]):
            target_charge[i] = np.clip(target_charge[i-1] + targets_in_sunlight[i] * charge_power * dt/3600, 0, bat_cap)
            target_charge[i] = np.clip(target_charge[i-1] + (1 - targets_in_sunlight[i]) * (sps_power[i] - hib_power) * dt/3600, 0, bat_cap)
        
#        for i in range(1, targets.shape[0]):
#            for j in range(targets.shape[1]):
#                
#                if targets_in_sunlight[i, j]:
#                    target_charge[i, j] = min(bat_cap, target_charge[i-1, j] + charge_power * dt/3600)
#                    
#                else:
#                    sps_power = 0                
#                    for k in range(sats.shape[1]):
#                        if contact[i, j, k]:
##                            n_targets = dict(zip(*np.unique(contact[i, :, k][np.logical_not(targets_in_sunlight[i])], return_counts=True)))[True]
#                            
#                            sps_power += eff[i, j, k] * sat_power / n_targets_t[i, k]
##                        else:
##                            n_targets = 0
#                            
##                        if j == 6 and i == 1:
##                            print(contact[i, :, k][np.logical_not(targets_in_sunlight[i])])
##                            print(contact[i, :, k], np.logical_not(targets_in_sunlight[i]))
##                            print(n_targets - n_targets_t[i, j], n_targets_t[i, k])
#                    if j == 0:
#                        print(sps_power - sps_power_t[i, j])
#                            
#                    target_charge[i, j] = min(bat_cap, max(0, target_charge[i-1, j] + (sps_power - hib_power) * dt/3600))
        
        
        
        return target_charge        
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    sma = cte.h_crit
    inc = np.radians(50)
    
    n_planes = 4
    n_sats_plane = 3
    
    coverage = Coverage()
    charge = coverage.fitness([sma, inc, np.pi/2, 2*np.pi/3])
        
    plt.figure(2)
    for i in range(coverage.targets.shape[1]):
        plt.subplot(2, 4, i + 1)
        plt.plot(coverage.sim_time, charge[:, i])
