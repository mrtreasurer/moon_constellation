# -*- coding: utf-8 -*-

import numpy as np


def create_targets(coors, sim_time, omega_moon):
    pos0 = np.zeros(coors.shape)
    for i, co in enumerate(coors):
        r_m = co[0]
        lat = co[1]
        lon = co[2]
        
        pos0[i] = [r_m * np.cos(lon)*np.cos(lat),
                   r_m * np.sin(lon)*np.cos(lat),
                   r_m * np.sin(lat)]
    
    def rot_mat(alpha):
        rot = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])
        
        return rot
    
    rot_mat_vec = np.vectorize(rot_mat, signature='()->(3,3)')
    
    rot = rot_mat_vec(sim_time*omega_moon)
    
    states = np.stack([np.matmul(rot, pos0[i].reshape(3, 1)) for i in range(pos0.shape[0])])

    return np.transpose(states[:, :, :, 0], (1, 0, 2))
