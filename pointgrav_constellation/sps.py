# -*- coding: utf-8 -*-
import numpy as np
import pykep as pk

from propagation import dx_dt
from integration import rk4
from periodic_orbits.constants import mu_sh, r_sh

import constants as cte


class Satellite:
    def __init__(self, ident, time, pos, kep0):
        self.ident = ident
        
        self.time = time
        self.pos = pos
        
        self.kep = kep0
        
        self.period = 2*np.pi * np.sqrt(self.kep[0]**3/mu_sh)
        
        self.targets_in_range = {}
        self.n_targets_in_range = np.zeros(self.pos.shape[0])
        
        self.sunlight = np.zeros(self.pos.shape[0])
        
        self.laser_power = 8e3  # W
        
        self.n_las = 0.4
        self.n_geom = 0.8
        self.point_acc = 0.1e-6  # rad
        
    def in_sunlight(self, epoch, pos_sun):
        pos = self.pos[epoch]
        
        cos_psi = np.dot(pos_sun, pos)/(np.linalg.norm(pos)*np.linalg.norm(pos_sun))
        a = pos * np.sqrt(1 - cos_psi**2)
        
        sunlight = True
        if cos_psi < 0 and a < r_sh:
            sunlight = False
            
        self.sunlight[epoch] = sunlight
        
    def in_range(self, i, tar):
        self.n_targets_in_range[i] += 1
        
        if i in self.targets_in_range:
            self.targets_in_range[i].append(tar)
            
        else:
            self.targets_in_range[i] = [tar]
            
            
    def link_eff(self, tar, epoch):
        r = np.linalg.norm(tar.pos[i] - self.pos[i])
        
        n_trans = min((1/3)**2, (tar.rec_r/(self.point_acc*r))**2)
        
        return self.n_las * n_trans * self.n_geom * tar.rec_n


rge = 2.2e6  # m
elev = np.radians(10)  # rad

h_crit = np.sqrt(r_sh**2 + rge**2 + 2*r_sh*rge*np.sin(elev))

sma = h_crit
alt = sma - r_sh

n_planes = 4
n_sats_plane = 3
n_sats = n_planes * n_sats_plane

inc = np.radians(60)
e = 0
aop = 0
raan_list = np.linspace(0, 2*np.pi, n_planes + 1)[:-1]
ta = np.linspace(0, 2*np.pi, n_sats_plane + 1)[:-1]

sats_pos = np.zeros((n_sats_plane, 6))
sats_kep = np.copy(sats_pos)        

for j, s in enumerate(range(n_sats_plane)):
    sats_pos[j] = np.concatenate(pk.core.par2ic([sma, e, inc, 0, aop, ta[j]], mu_sh))
    sats_kep[j] = np.array([sma, e, inc, 0, aop, ta[j]])        

sat_hist = np.zeros((n_sats, cte.sim_time.shape[0], 3))

sats = []
for i in range(n_sats_plane):
    sat_hist[i] = rk4(cte.sim_time, sats_pos[i], dx_dt)[:, 0:3]
    
    for j in range(1, n_planes):
        raan = raan_list[j]
        rot = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan), np.cos(raan), 0],
                        [0, 0, 1]])
        
        sat_hist[n_sats_plane*j + i] = np.transpose(np.dot(rot, np.transpose(sat_hist[i])))
        
for i, sat in enumerate(sat_hist):
    sats.append(Satellite(i, cte.sim_time, sat, [sma, e, inc, raan_list[i//n_sats_plane], aop, ta[i%n_sats_plane]]))


#if __name__ == "__main__":
#    print(mu_sh, r_sh)
#    
#    print(np.linalg.norm(sats_pos[0, 3:6]))
#    print(np.sqrt(mu_sh/sma))