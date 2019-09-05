#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:02:47 2019

@author: bert
"""

import numpy as np
import pykep as pk

import constants as cte


x = -1.311519120505e-2
y = 5.435394815081e-4
z = 0.

pos = np.array([x, y, z])

acc0 = pk.util.gravity_spherical_harmonic(np.array([pos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

xpos = pos + np.array([0.01/cte.lu, 0, 0])
accx = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

xpos = pos + np.array([0, 0.01/cte.lu, 0])
accy = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

xpos = pos + np.array([0, 0, 0.01/cte.lu])
accz = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

dx = -(acc0 - accx) / 0.01 * cte.lu
dy = -(acc0 - accy) / 0.01 * cte.lu
dz = -(acc0 - accz) / 0.01 * cte.lu