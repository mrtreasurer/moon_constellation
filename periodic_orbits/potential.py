#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 20:50:01 2019

@author: bert
"""

import numpy as np

from definitions import legendre, cos_lambda, sin_lambda


def potential(state, r_planet, mu, c, s, max_degree, max_order):   
    r = np.linalg.norm(state)
    
    u = mu/r
    
    if max_degree >= 2:
        sin_theta = state[2]/r
                
        if max_order >= 1:
            p = legendre(max_degree, sin_theta)
            cos_l = cos_lambda(state, max_order)
            sin_l = sin_lambda(state, max_order)
    
        re_r = r_planet/r
        re_rn = r_planet/r    
    
        inf_sum = 0
    
        for n in range(2, max_degree + 1):
            re_rn *= re_r
            
            inf_sum += c[n, 0] * re_rn * p[n, 0]        
            
            if max_order >= 1:
                for m in range(1, min(n + 1, max_order + 1)):
                    inf_sum += re_rn * p[n, m] * (c[n, m] * cos_l[m] + s[n, m] * sin_l[m])
                            
        u *= inf_sum
    
    return u


def norm_factor(n, m):
    def factorial(x):
        f = 1
        for y in range(1, x + 1):
            f *= y
            
        return f
    
    factor = np.sqrt(factorial(n - m) * (2*n - 1) * (2 - int(m == 0)) / factorial(m + n))
    
    return factor
        

if __name__ == "__main__":
    import pykep as pk
    
    import constants as cte
    
    
    r_planet, mu_planet, cos, sin, max_degree, max_order = pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/jgl_150q1.txt")
    
    x0 = -4.498948742093e-03
    y0 = -1.731769313131e-03
    z0 = 0.
    
    state = np.array([x0, y0, z0]) * cte.lu
    
    u = potential(state, r_planet/1e3, mu_planet/1e9, cos, sin, 50, 50)

    print(u)
