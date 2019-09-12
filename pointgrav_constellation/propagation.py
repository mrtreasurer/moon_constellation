# -*- coding: utf-8 -*-

import numpy as np
import pykep as pk

import periodic_orbits.constants as cte

#from numba import jit


#@jit
def dx_dt(t, state):

    r = np.linalg.norm(state[0:3])

    dg = - cte.mu_sh/r**3 * state[0:3]

    dx = np.concatenate((state[3:6], dg))

    return dx