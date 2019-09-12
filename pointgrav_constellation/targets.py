# -*- coding: utf-8 -*-

import numpy as np

from periodic_orbits.constants import r_sh

import constants as cte


class Target:
    def __init__(self, ident, pos, time):
        self.ident = ident
        self.pos = pos
        self.time = time
            
        self.sats_in_range = {}
        self.n_sats_in_range = np.zeros(time.shape)
        
        self.sunlight = self.pos[:, 0] >= 0
        
        self.op_power = 150  # W
        self.battery_cap = 500  # Whr
        self.hibernation_power = 42  # W
        self.charge_power = 240  # W
        
        self.charge = np.zeros(time.shape)
        self.charge[0] = self.battery_cap
        
        self.rec_n = 0.5
        self.rec_r = np.sqrt(1/np.pi)
        
    def in_range(self, i, sat):
        self.n_sats_in_range[i] += 1
        
        if i in self.sats_in_range:
            self.sats_in_range[i].append(sat)
            
        else:
            self.sats_in_range[i] = [sat]
            
    def update_charge(self, epoch, sps_power):        
        if self.sunlight[epoch]:            
            self.charge[epoch] = self.charge[epoch - 1] \
                + (self.charge_power - self.op_power) * (self.time[epoch] - self.time[epoch - 1])/3600
        
        else:
            self.charge[epoch] = self.charge[epoch - 1] \
                + (min(sps_power, self.charge_power) - self.hibernation_power) * (self.time[epoch] - self.time[epoch - 1])/3600
                
        if self.charge[epoch] > self.battery_cap:
            self.charge[epoch] = self.battery_cap
            
        elif self.charge[epoch] < 0:
            self.charge[epoch] = 0


lats = np.radians([58.1, 43.914, 30.76515, 14, -70.9, -45.5])
lons = np.radians([309.1, 25.148, 20.19069, 303.5, 22.8, 177.6])

targets_pos = []

for lon, lat in zip(lons, lats):
    pos = [r_sh * np.cos(lon)*np.cos(lat),
           r_sh * np.sin(lon)*np.cos(lat),
           r_sh * np.sin(lat)]
    
    targets_pos.append(pos)

targets_pos = np.array(targets_pos)

target_hist = np.zeros((targets_pos.shape[0], cte.sim_time.shape[0], 3))  
target_hist[:, 0] = targets_pos

for i in range(targets_pos.shape[0]):
    for j in range(1, cte.sim_time.shape[0]):
        rot = np.array([[np.cos(cte.sim_time[j] * cte.omega_moon), -np.sin(cte.sim_time[j] * cte.omega_moon), 0],
                        [np.sin(cte.sim_time[j] * cte.omega_moon), np.cos(cte.sim_time[j] * cte.omega_moon), 0],
                        [0, 0, 1]])
        
        target_hist[i, j] = np.dot(rot, targets_pos[i])
        
targets = []
for i, tar in enumerate(target_hist):
    targets.append(Target(i, tar, cte.sim_time))
            

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, 2*np.pi, 20)
    x = r_sh * np.outer(np.cos(u), np.sin(v))
    y = r_sh * np.outer(np.sin(u), np.sin(v))
    z = r_sh * np.outer(np.ones(np.size(u)), np.cos(v))
    
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(x, y, z, color='b')
    ax.plot(targets_pos[:, 0], targets_pos[:, 1], targets_pos[:, 2], 'mo')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    for i, tar in enumerate(targets):
        plt.figure(2 + i)
        plt.plot(tar.time, tar.pos[:, 0]/r_sh)
        plt.plot(tar.time, tar.sunlight)