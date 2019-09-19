#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:02:47 2019

@author: bert
"""

import numpy as np
import pykep as pk

import constants as cte

from potential import potential
from d2pot import d2u_dx2


x = -1.311519120505e-2
y = 5.435394815081e-4
z = 0.

step = 0.001

pos = np.array([x, y, z])

acc0 = pk.util.gravity_spherical_harmonic(np.array([pos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)[0]

point_g0 = cte.mu_n/np.linalg.norm(pos)
pot0 = point_g0 + potential(pos, cte.r_n, cte.mu, cte.cos, cte.sin, cte.degree, cte.order)

d2u0 = d2u_dx2(pos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

xpos = pos + np.array([step/cte.lu, 0, 0])
accx = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)[0]

point_g0 = cte.mu_n/np.linalg.norm(xpos)
potx = point_g0 + potential(xpos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

xpos = pos + np.array([0, step/cte.lu, 0])
accy = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)[0]

point_g0 = cte.mu_n/np.linalg.norm(xpos)
poty = point_g0 + potential(xpos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)


xpos = pos + np.array([0, 0, step/cte.lu])
accz = pk.util.gravity_spherical_harmonic(np.array([xpos]), cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)[0]

point_g0 = cte.mu_n/np.linalg.norm(xpos)
potz = point_g0 + potential(xpos, cte.r_n, cte.mu_n, cte.cos, cte.sin, cte.degree, cte.order)

dadx = -(acc0 - accx) / step * cte.lu - d2u0[0]
dady = -(acc0 - accy) / step * cte.lu - d2u0[1]
dadz = -(acc0 - accz) / step * cte.lu - d2u0[2]

dpdx = -(pot0 - potx) / step * cte.lu - acc0[0]
dpdy = -(pot0 - poty) / step * cte.lu - acc0[1]
dpdz = -(pot0 - potz) / step * cte.lu - acc0[2]