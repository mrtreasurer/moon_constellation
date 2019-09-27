import numpy as np
import pygmo as pg

from matplotlib import pyplot as plt

import constants as cte

from optim_class import Coverage, initiate

# def optimise():
sim_time, sun_pos, targets = initiate()

coverage = Coverage(sim_time, sun_pos, targets, cte.r_m, cte.mu_m, cte.dt, cte.min_elev, cte.max_sat_range, cte.ecc, cte.aop, cte.tar_battery_cap, cte.tar_charge_power, cte.sat_las_power, cte.tar_hib_power, cte.sat_point_acc, cte.tar_r_rec, cte.sat_n_las, cte.sat_n_geom, cte.tar_n_rec, cte.sat_wavelength, cte.sat_r_trans)

prob = pg.problem(coverage)
algo_class = pg.gaco(gen=10)
# algo_class.set_bfe(pg.bfe())
algo = pg.algorithm(algo_class)

pop = pg.population(prob, 100)

algo.set_verbosity(1)
pop = algo.evolve(pop)

print(pop.champion_x, pop.champion_f)
# return pop.champion_x, pop.champion_f