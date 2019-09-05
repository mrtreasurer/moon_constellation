#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:08:01 2019

@author: bert
"""

import numpy as np

from numba import jit
 

@jit
def legendre(n_max, x):
#    cos_x = np.sqrt(1 - x**2)
    
    p = np.zeros((n_max + 1, n_max + 1))

    p[0, 0] = 1
    p[1, 0] = x
    p[1, 1] = np.sqrt(3)
    
    for n in range(2, n_max + 1):
        nn1 = np.sqrt((2*n + 1)/(2*n - 1))
        nn2 = np.sqrt((2*n + 1)/(2*n - 3))
        n1n1 = 1/(2*n - 1) * np.sqrt((2*n + 1)/(2*n))
        
        p[n, 0] = (2*n - 1)/n * nn1 * x*p[n - 1, 0] - (n - 1)/n * nn2 * p[n - 2, 0]
        p[n, n] = (2*n - 1) * p[n - 1, n - 1] * n1n1
        
        for m in range(1, n):          
            n1m = np.sqrt((n - m) * (2*n + 1) / ((n + m) * (2*n - 1)))
            n2m = np.sqrt((n - m) * (n - m - 1) * (2*n + 1) / ((n + m) * (n + m - 1) * (2*n - 3)))
            
            p[n, m] = (2*n - 1)/(n - m) * n1m * x*p[n - 1, m] - (n + m - 1)/(n - m) * n2m * p[n - 2, m]
            
    return p


@jit
def cos_lambda(state, m_max): 
    cos_list = np.ones(m_max + 1)
    cos_list[1] = state[0]/np.linalg.norm(state[0:2])
    
    cos_l = cos_list[1]
    
    for m in range(2, m_max + 1):
        cos_list[m] = 2 * cos_l * cos_list[m - 1] - cos_list[m - 2]
    
    return cos_list
        

@jit
def sin_lambda(state, m_max):
    sin_list = np.zeros(m_max + 1)
    sin_list[1] = state[1]/np.linalg.norm(state[0:2])
    
    cos_l = state[0]/np.linalg.norm(state[0:2])
    
    for m in range(2, m_max + 1):
        sin_list[m] = 2 * cos_l * sin_list[m - 1] - sin_list[m - 2]
    
    return sin_list


def distances(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    rho = np.sqrt((x - 1)**2 + y**2 + z**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    return r, rho


def d2_gamma(pos, mu, d2u):
    dummy, rho = distances(pos)
    
    a = np.ones((3, 3))
    
    a[0, 0] = 1 - (1 - mu) * (rho**2 - 3*(pos[0] - 1)**2)/rho**5
    a[1, 1] = 1 - (1 - mu) * (rho**2 - 3*pos[1]**2)/rho**5
    a[2, 2] = - (1 - mu) * (rho**2 - 3*pos[2]**2)/rho**5
    
    a[1, 0] = a[0, 1] = (1 - mu) * 3*(pos[0] - 1)*pos[1]/rho**5
    a[2, 0] = a[0, 2] = (1 - mu) * 3*(pos[0] - 1)*pos[2]/rho**5
    a[2, 1] = a[1, 2] = (1 - mu) * 3*pos[1]*pos[2]/rho**5
    
    d2g = a + d2u
    
    return d2g


def jacobian(state, mu, u):
    r, rho = distances(state[0:3])

    gamma = (state[0] - 1 + mu)**2/2 + state[1]**2/2 + (1 - mu)/rho + (mu/r + u)
    
    jac = 2*gamma - sum(state[3:6]**2)
    
    return jac


def d_jac(state, mu, acc):
    r, rho = distances(state[0:3])
    
    dc = np.zeros(6)
    
    dc[0] = 2 * ((state[0] - 1 + mu) - (1 - mu)*(state[0] - 1)/rho**3 + acc[0])
    dc[1] = 2 * (state[1] - (1 - mu)*state[1]/rho**3 + acc[1])
    dc[2] = 2 * (- (1 - mu)*state[2]/rho**3 + acc[2])
    dc[3] = - 2*state[3]
    dc[4] = - 2*state[4]
    dc[5] = - 2*state[5]
    
    return dc
