# -*- coding: utf-8 -*-

import pygmo as pg

from optim_class import Coverage, initiate

def optimize():
    sim_time, sun_pos, targets = initiate()

    # 1 - Instantiate a pygmo problem constructing it from a UDP
    # (user defined problem).
    prob = pg.problem(Coverage(sim_time, sun_pos, targets))

    # 2 - Instantiate a pagmo algorithm
    algo = pg.algorithm(pg.sade(gen=100))

    # 3 - Instantiate an archipelago with 16 islands having each 20 individuals
    archi = pg.archipelago(1, algo=algo, prob=prob, pop_size=7)

    # 4 - Run the evolution in parallel on the 16 separate islands 10 times.
    archi.evolve(1)  

    # 5 - Wait for the evolutions to be finished
    archi.wait()

    # 6 - Print the fitness of the best solution in each island
    results = ""

    for isl in archi:
        print(isl.get_population().champion_x, isl.get_population().champion_f)
        results += "\t".join(isl.get_population().champion_x + [isl.get_populatin().champion_f])
        
    with open("results.txt", "w") as f:
        f.write(results)