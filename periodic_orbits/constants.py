#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:02:30 2019

@author: bert
"""

import numpy as np
import pykep as pk

mu_m = 4.902801076e12  # m3 s-2
mu_e = 3.986004415e14  # m3 s-2


mu = mu_m / (mu_e + mu_m)

r_em = 384400000  # m
lu = r_em  # m
tu = np.sqrt(lu**3/(mu_e+mu_m))  # s

r_sh, mu_sh, cos, sin, max_degree, max_order = pk.util.load_gravity_model("/home/bert/miniconda3/envs/pykep/lib/python3.7/site-packages/pykep/util/gravity_models/Moon/jgl_150q1.txt")

r_n = r_sh / lu
mu_n = mu_sh / lu**3 * tu**2

degree = 10
order = 10