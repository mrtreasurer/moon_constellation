# -*- coding: utf-8 -*-

import numpy as np

import constants as cte
import definitions as d
import sps
import targets as tgt


def initiate():
    sim_time = np.arange(0, cte.synodic_period + cte.dt, cte.dt)
    #sim_time = np.arange(0, 2*np.pi*np.sqrt(cte.h_crit**3/cte.mu_m), cte.dt)
    
    sun_pos = d.sun_loc(sim_time, cte.omega_earth, cte.sun_pos0)
    
    targets = tgt.create_targets(cte.target_coors, sim_time, cte.omega_moon)
    
    return sim_time, sun_pos, targets


class Coverage:
    def __init__(self, time, sun, targets, r_m, mu_m, dt, min_elev, max_sat_range, ecc, aop, tar_battery_cap, tar_charge_power, sat_las_power, hib_power, sat_point_acc, tar_r_rec, sat_n_las, sat_n_geom, tar_n_rec, wavelength, r_trans):
        self.sim_time = time
        
        self.sun = sun
        
        self.targets = targets

        self.r_m = r_m
        self.min_elev = min_elev
        self.max_sat_range = max_sat_range

        self.tar_battery_cap = tar_battery_cap
        self.tar_charge_power = tar_charge_power

        self.dt = dt

        self.sat_las_power = sat_las_power

        self.hib_power = hib_power

        self.ecc = ecc
        self.aop = aop
        self.mu_m = mu_m

        self.sat_point_acc = sat_point_acc
        self.tar_r_rec = tar_r_rec
        self.sat_n_las = sat_n_las
        self.sat_n_geom = sat_n_geom
        self.tar_n_rec = tar_n_rec

        self.wavelength = wavelength
        self.r_trans = r_trans
                
    def fitness(self, x):
        sma = x[0]
        inc = x[1]
        # plane_sep = x[2]
        # sat_sep = x[3]
        
        # n_planes = int(2*np.pi // plane_sep)
        # n_sats_plane = int(2*np.pi // sat_sep)

        n_planes = int(x[2])
        n_sats_plane = int(x[3])
                
        sats = self.create_constellation(sma, inc, n_planes, n_sats_plane)
        
        charge = self.propagate_constellation(sats, sma)[0]

        min_charge = np.min(charge)
        
        fitness = (n_planes * n_sats_plane) + 1/float(np.mean(charge)) + (10 * (min_charge < 0.1*self.tar_battery_cap))
        
        return [fitness]

    def fitness_2d(self, x, n_planes, n_sats_plane):
        sma = x[0]
        inc = x[1]
        # plane_sep = x[2]
        # sat_sep = x[3]
        
        # n_planes = int(2*np.pi // plane_sep)
        # n_sats_plane = int(2*np.pi // sat_sep)

        # n_planes = int(x[2])
        # n_sats_plane = int(x[3])
                
        sats = self.create_constellation(sma, inc, n_planes, n_sats_plane)
        
        charge = self.propagate_constellation(sats, sma)[0]

        min_charge = np.min(charge)
        
        fitness = (n_planes * n_sats_plane) + 1/float(np.mean(charge)) + (10 * (min_charge < 0.1*self.tar_battery_cap))
        
        return fitness

    def get_nix(self):
        return 2
    
    def get_bounds(self):
        return ([self.r_m, np.pi/4, 1, 1], [self.max_sat_range, np.pi/2, 5, 5])

    def create_constellation(self, sma, inc, n_planes, n_sats_plane):        
        raan = np.repeat(np.linspace(0, 2*np.pi, n_planes + 1)[:-1], n_sats_plane)
        ta = np.tile(np.linspace(0, 2*np.pi, n_sats_plane + 1)[:-1], n_planes)
        
        kep = np.empty((n_planes * n_sats_plane, 6))
        kep[:, 0] = sma
        kep[:, 1] = self.ecc
        kep[:, 2] = inc
        kep[:, 3] = raan
        kep[:, 4] = self.aop
        kep[:, 5] = ta
        
        sats = sps.create_sats(kep, self.mu_m, self.sim_time)
        
        return sats
    
    def propagate_constellation(self, sats, sma):
        cos_min_elev = np.cos(self.min_elev)
        sin_rho = self.r_m / sma
        sin_max_nadir = sin_rho * cos_min_elev
        cos_max_lambda = np.cos(np.pi/2 - self.min_elev - np.arcsin(sin_max_nadir))
        
        dist_vec = np.empty((self.targets.shape[0], self.targets.shape[1], sats.shape[1], 3))
        
        for i in range(sats.shape[1]):
            dist_vec[:, :, i] = np.transpose(np.stack([np.subtract(sats[:, i], self.targets[:, j]) for j in range(self.targets.shape[1])]), (1, 0, 2))
            
        dist = np.linalg.norm(dist_vec, axis=3)
        
        einsum_targets = np.sqrt(np.einsum("ijm,ijm->ij", self.targets, self.targets))
        einsum_sats = np.sqrt(np.einsum("ijm,ijm->ij", sats, sats))
        
        cos_lambda = np.einsum("ijk,imk->imj", sats, self.targets) / \
        (np.concatenate([einsum_sats[:, None, :]]*self.targets.shape[1], axis=1) * \
         np.concatenate([einsum_targets[:, :, None]]*sats.shape[1], axis=2))
        
        contact = (cos_lambda > cos_max_lambda) * (dist < self.max_sat_range)
        
        eff = d.link_eff(dist, self.sat_point_acc, self.tar_r_rec, self.sat_n_las, self.sat_n_geom, self.tar_n_rec, self.wavelength, self.r_trans)
        
        einsum_sun = np.sqrt(np.einsum("ij,ij->i", self.sun, self.sun))
        
        cos_sun_angle_t = np.einsum("ik,ijk->ij", self.sun, self.targets) / \
        (np.concatenate([einsum_sun[:, None]]*self.targets.shape[1], axis=1) * 
         einsum_targets)
        
        targets_in_sunlight = cos_sun_angle_t > 0
           
    #    cos_sun_angle_s = np.einsum("ik,ijk->ij", sun, sats) / \
    #    (np.concatenate([einsum_sun[:, None]]*sats.shape[1], axis=1) * 
    #     einsum_sats)
    #    
    #    sats_in_sunlight = (cos_sun_angle_s > 0) + (np.sqrt(1 - cos_sun_angle_s**2) > r_m/sma)
    
        n_targets = np.sum(contact * np.logical_not(targets_in_sunlight)[:, :, None], axis=1)
        sps_power = np.sum((eff * self.sat_las_power / np.clip(np.transpose([n_targets]*eff.shape[1], (1, 0, 2)), 1, np.inf)) * contact, axis=2) * np.logical_not(targets_in_sunlight)
        
        target_charge = np.empty((self.targets.shape[0:2]))
        target_charge[0] = self.tar_battery_cap
        
        for i in range(1, self.targets.shape[0]):
            target_charge[i] = np.clip(target_charge[i-1] + np.logical_not(targets_in_sunlight[i]) * (sps_power[i] - self.hib_power) * self.dt/3600 + targets_in_sunlight[i] * self.tar_charge_power * self.dt/3600, 0, self.tar_battery_cap)
            # target_charge[i] = np.clip(target_charge[i-1], 0, bat_cap)        
        
        return target_charge, dist, eff, contact, targets_in_sunlight
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    sma = 2.54279861e6
    inc = 1.02596289

    n_planes = 4
    n_sats_plane = 1
    
    sim_time, sun, targets_pos = initiate()
    
    coverage = Coverage(sim_time, sun, targets_pos, cte.r_m, cte.mu_m, cte.dt, cte.min_elev, cte.max_sat_range, cte.ecc, cte.aop, cte.tar_battery_cap, cte.tar_charge_power, cte.sat_las_power, cte.tar_hib_power, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec, cte.sat_wavelength, cte.sat_r_trans)
            
    sats = coverage.create_constellation(sma, inc, n_planes, n_sats_plane)
    
    print(coverage.fitness([sma, inc, n_planes, n_sats_plane]))
    charge, dist, eff, contact, targets_in_sunlight = coverage.propagate_constellation(sats, sma)

    plt.figure(1)
    ax = plt.axes(projection='3d')

    sat_period = 2*np.pi * np.sqrt(sma**3/cte.mu_m)

    for i in range(sats.shape[1]):
        arr1 = sats[:, i, 0][sim_time < sat_period]/1e3
        arr2 = sats[:, i, 1][sim_time < sat_period]/1e3
        arr3 = sats[:, i, 2][sim_time < sat_period]/1e3

        np.savetxt(f"data/figure1_sats_{i}.csv", np.column_stack((arr1, arr2, arr3)), delimiter=",")

        ax.plot(arr1, arr2, arr3, 'b')
        ax.plot([arr1[0]], [arr2[0]], [arr3[0]], 'bo')
        
    for i in range(targets_pos.shape[1]):
        arr1 = targets_pos[::100, i, 0]/1e3
        arr2 = targets_pos[::100, i, 1]/1e3
        arr3 = targets_pos[::100, i, 2]/1e3

        np.savetxt(f"data/figure1_targets_{i}.csv", np.column_stack((arr1, arr2, arr3)), delimiter=",")

        ax.plot(arr1, arr2, arr3, 'g')
        ax.plot([arr1[0]], [arr2[0]], [arr3[0]], 'go')

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")

    plt.savefig("data/figure1")
        
    plt.figure(2)
    for i in range(targets_pos.shape[1]):
        plt.subplot(2, 3, i + 1)

        arr1 = sim_time[np.logical_not(targets_in_sunlight[:, i])]/24/3600
        arr2 = charge[:, i][np.logical_not(targets_in_sunlight[:, i])]/cte.tar_battery_cap*100

        np.savetxt(f"data/figure2_{i}.csv", np.column_stack((arr1, arr2)))

        plt.plot(arr1, arr2)

        plt.xlabel("Time [days]")
        plt.ylabel("Battery charge [%]")

    plt.savefig("data/figure2")

    plt.figure(3)
    plt.plot(sim_time[contact[:, 0, 0]], eff[:, 0, 0][contact[:, 0, 0]], '.')

    # plt.show()
