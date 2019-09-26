# -*- coding: utf-8 -*-

import numpy as np

r_em = 3.84401e8  # m
r_es = 149597870.66e3  # m

r_m = 1738e3  # m

sun_pos0 = np.array([r_es, 0, 0])

mu_m = 4906.355427e9  # m3/s2
mu_e = 3.98600441e14  # m3/s2
mu_s = 1.327178e20  # m3/s2

moon_period = 2*np.pi * np.sqrt(r_em**3/(mu_e + mu_m))  # s
earth_period = 2*np.pi * np.sqrt(r_es**3/(mu_s + mu_m + mu_e))  # s

synodic_period = moon_period * earth_period / (earth_period - moon_period)

dt = 60  # s

omega_moon = 2*np.pi / moon_period  # rad/s
omega_earth = 2*np.pi / earth_period # rad/s

# (lat [deg], lon [deg])
_target_coors_deg = np.array([[58.1, 309.1],
                              [43.914, 25.148],
                              [30.76515, 20.19069],
                              [14, 303.5],
                              [-70.9, 22.8],
                              [-45.5, 177.6]])

# (lat [rad], lon [rad])
target_coors = np.column_stack((r_m*np.ones(_target_coors_deg.shape[0]), _target_coors_deg * np.pi/180))

max_sat_range = 2491e3  # m
min_elev = np.radians(10)  # rad

h_crit = np.sqrt(r_m**2 + max_sat_range**2 + 2*r_m*max_sat_range*np.sin(min_elev))

ecc = 0
aop = 0

sat_point_acc = 0.1e-6  # rad
sat_n_las = 0.4
sat_n_geom = 0.8
sat_las_power = 7.79e3  # W
sat_wavelength = 1070e-9  # m
sat_r_trans = np.sqrt(max_sat_range * sat_wavelength / np.pi)  # m

# tar_op_power = 150  # W
tar_battery_cap = 500  # Whr
tar_hib_power = 42  # W
tar_charge_power = 240 - 150  # W
       
tar_n_rec = 0.5
tar_r_rec = np.sqrt(1.83/np.pi)  # m

r_beam = tar_r_rec + np.sqrt(1 + (sat_wavelength * max_sat_range / (np.pi * sat_r_trans**2))**2)