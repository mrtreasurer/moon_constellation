# -*- coding: utf-8 -*-

import numpy as np


sim_end = 24*3600 * 29.53 # s
dt = 60  # s

sim_time = np.arange(0, sim_end, dt)

omega_moon = 2*np.pi / sim_end  # rad/s