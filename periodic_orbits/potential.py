#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:35:51 2019

@author: bert
"""

import numpy as np

from numba import jit

from definitions import legendre, cos_lambda, sin_lambda

@jit
def potential(pos, planet_r, mu, c, s, degree, order):
    pot = mu/planet_r
    
    inf_sum = 0
    
    r = np.linalg.norm(pos)    
    sin_phi = pos[2]/r
    
    p = legendre(degree, sin_phi)
    cos_l = cos_lambda(pos, order)
    sin_l = sin_lambda(pos, order)
    
    rp_r = planet_r/r
    rp_rn = rp_r
    
    for n in range(2, degree + 1):
        rp_rn *= rp_r
        
        inf_sum += c[n, 0] * rp_rn * p[n, 0]
        
        for m in range(1, min(order + 1, n + 1)):
            inf_sum += rp_rn * (c[n, m] * cos_l[m] + s[n, m] * sin_l[m]) * p[n, m]
            
    pot *= inf_sum
    
    return pot


if __name__ == "__main__":
    import constants as cte
    
    coor = np.array([-1.311519120505e-2,
                     5.435394815081e-4,
                     0.])
    
#    for n in range(5):
#        for m in range(n + 1):
#            pt = potential(coor, cte.r_n, cte.mu_n, cte.cos, cte.sin, n, m)
#            print(f"{n}:{m} {pt}")

    pt10 = potential(coor, cte.r_n, cte.mu_n, cte.cos, cte.sin, 50, 50)
    print((pt10 - 1.9064881091046537e-05)/1.9064881091046537e-05)
