from time_optimisation import *
import traj_gen
import numpy as np

from traj_gen import poly_trajectory as pt

import time

waypoints = np.array([
				[1, 3, 5], 
				[2, 8, 11],
				[4, 10, 17],
				[8, 12, 16],
		])

starter_knots = get_knots(waypoints, scale = 10)
final_knots = optimise_knots(waypoints)

print("Initial - ")
print(starter_knots)

print("Final - ")
print(final_knots)