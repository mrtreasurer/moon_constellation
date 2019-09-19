# -*- coding: utf-8 -*-

import numpy as np


def sun_loc(sim_time, omega, pos0):
    def rot_mat(alpha):
        rot = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 1]])
        
        return rot
    
    rot_mat_vec = np.vectorize(rot_mat, signature='()->(3,3)')
    
    rot = rot_mat_vec(sim_time*omega)
    
    state = np.stack(np.matmul(rot, pos0.reshape(3, 1)))
    
    return state[:, :, 0]


def link_eff(dist, point_acc, r_rec, n_las, n_geom, n_rec):
    n_trans = np.minimum((1/3)**2, (r_rec/(point_acc*dist))**2)
    
    return n_las * n_trans * n_geom * n_rec
