import numpy as np

from matplotlib import pyplot as plt

import constants as cte

from optim_class import Coverage, initiate

sat_period = 2*np.pi * np.sqrt(cte.optim_sma**3/cte.mu_m)

sim_time, sun, targets_pos = initiate(cte.target_coors, cte.sun_pos0, cte.omega_earth, cte.omega_moon, cte.dt, cte.synodic_period)

coverage = Coverage(sim_time, sun, targets_pos, cte.r_m, cte.mu_m, cte.dt, cte.min_elev, cte.max_sat_range, cte.ecc, cte.aop, cte.tar_battery_cap, cte.tar_charge_power, cte.sat_las_power, cte.tar_hib_power, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec, cte.sat_wavelength, cte.sat_r_trans)

inc_range = np.arange(np.degrees(cte.optim_inc) - 20, np.degrees(cte.optim_inc) + 20, 1)*np.pi/180

inc_results = np.zeros((inc_range.shape[0], 2))
for j, inc in enumerate(inc_range):
    print(f"\rInlination: {round(np.degrees(inc), 2)} deg", end="")

    sats = coverage.create_constellation(cte.optim_sma, inc, cte.optim_n_planes, cte.optim_n_sats_plane)
    charge, dist, eff, contact, targets_in_sunlight, n_targets = coverage.propagate_constellation(sats, cte.optim_sma)

    inc_results[j] = [np.mean(charge), np.min(charge)]

np.savetxt("data/inc_sensitivity.csv", inc_results, delimiter=",")
print("\rInclination analysis finished")

sma_range = np.arange(cte.optim_sma - 4e5, cte.optim_sma + 6e5, 1e4)

sma_results = np.zeros((sma_range.shape[0], 2))
for i, sma in enumerate(sma_range):
    print(f"\rSemi-major Axis: {round(sma/1e3, 2)} km", end="")

    sats = coverage.create_constellation(sma, cte.optim_inc, cte.optim_n_planes, cte.optim_n_sats_plane)
    charge, dist, eff, contact, targets_in_sunlight, n_targets = coverage.propagate_constellation(sats, sma)

    sma_results[i] = [np.mean(charge), np.min(charge)]

np.savetxt("data/sma_sensitivity.csv", sma_results, delimiter=",")
print("\rSemi-major Axis analysis finished")
