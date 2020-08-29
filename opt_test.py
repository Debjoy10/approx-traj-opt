import traj_gen
import numpy as np

from envs.traj_reward import *

from traj_gen import poly_trajectory as pt

import time

waypoints = np.array([
				[1, 3, 5], 
				[2, 8, 11],
				[4, 10, 17],
				[11, 12, 16],
		])

sknots = get_knots(waypoints, 2*len(waypoints))
ssnap = get_trajectory_snap_from_knots(waypoints, sknots)
fsnap = get_trajectory_snap(waypoints)

print("Initial Snap- {}".format(ssnap))
print("Final Snap- {}".format(fsnap))