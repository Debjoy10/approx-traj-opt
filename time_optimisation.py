import traj_gen
import numpy as np
import os
import contextlib
from traj_gen import poly_trajectory as pt

import time

def get_knots(waypoints, scale = 10):
	total_time = 0
	for wpa, wpb in zip(waypoints[1:], waypoints[:-1]):
		total_time += np.linalg.norm(wpa-wpb)    

	knots = np.zeros((len(waypoints)))
	knots[0] = 0
	for i, (wpa, wpb) in enumerate(zip(waypoints[1:], waypoints[:-1])):
		knots[i+1] = knots[i]+np.linalg.norm(wpa-wpb)/total_time
	
	return (knots*scale).astype(np.float)

def get_knots_from_Ti(Ti):
	current_t = 0
	knots = np.zeros(Ti.shape[0] + 1)
	for i in range(Ti.shape[0]):
		current_t += Ti[i]
		knots[i+1] = current_t
	return knots


def get_snap_from_knots(waypoints, knots, time_between_gates = 2, tdelta = 0.1, use_coeff_integral = False):
	dim = 3
	order = 8
	optimTarget = 'poly-coeff' #'end-derivative' 'poly-coeff'
	maxConti = 4
	objWeights = np.array([0, 0, 0, 0, 1])
	pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)

	# 2. Pin
	Xdot = np.array([0, 0, 0])
	Xddot = np.array([0, 0, 0])

	pin_ = {'t':0, 'd':1, 'X':Xdot,}
	pTraj.addPin(pin_)
	pin_ = {'t':0, 'd':2, 'X':Xddot,}
	pTraj.addPin(pin_)


	for i, wp in enumerate(waypoints):

		pin_ = {'t':knots[i], 'd':0, 'X':wp}
		pTraj.addPin(pin_)

	# solve
	pTraj.setDerivativeObj(objWeights)
	pTraj.solve()
	rng = np.linspace(0, len(waypoints)*time_between_gates, int((len(waypoints))*time_between_gates//tdelta))
	
	snap = pTraj.eval(rng, 4)
	return np.linalg.norm(snap)**2


def optimise_knots(waypoints):
	starter_knots = get_knots(waypoints, scale = 10)
	Tij = np.array([ti-tj for ti, tj in zip(starter_knots[1:], starter_knots[:-1])])
	m = Tij.shape[0]
	gi = np.eye(m) - ((np.ones([m, m]) - np.eye(m)).astype(float))/(m-1)

	h = 0.01
	stop_iter = 100
	lr = 1e-2
	iter_ = 0
	prev_snap = 1000

	while(iter_ < stop_iter):
		# To Suppress Print
		with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
			base_snap = get_snap_from_knots(waypoints, get_knots_from_Ti(Tij))

		if abs(prev_snap - base_snap) < 0.01:
			break

		delT = np.zeros_like(Tij)
		for i in range(m):
			T_i = Tij + h*gi[i]
			kn = get_knots_from_Ti(T_i)

			with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
				s = get_snap_from_knots(waypoints, kn)

			delT[i] = (s - base_snap)/h

		gradT = np.zeros_like(Tij)
		for i in range(m):
			gradT += (delT[i] * gi[i])

		Tij = Tij - lr*(gradT)
		iter_ += 1
		print("After iter{}, snap = {}".format(iter_, base_snap))
		prev_snap = base_snap

	return get_knots_from_Ti(Tij)
