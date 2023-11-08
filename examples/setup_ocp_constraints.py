'''
This code shows how to setup quickly an Optimal Control Problem (OCP) 
from a templated YAML file (with constraints)
'''

import numpy as np

from mim_robots.robot_loader import load_pinocchio_wrapper

from croco_mpc_utils.utils import load_yaml_file
from croco_mpc_utils.ocp_constraints import OptimalControlProblemClassicalWithConstraints
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical

import mim_solvers

# Read YAML config file
config = load_yaml_file('ocp_constraint.yml')

# Import robot model (pinocchio wrapper)
robot  = load_pinocchio_wrapper('iiwa')

# Setup the OCP using the wrapper
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])
ocp = OptimalControlProblemClassicalWithConstraints(robot, config).initialize(x0)

# Initialize OCP solver
solver = mim_solvers.SolverCSQP(ocp)
solver.max_qp_iters = 1000
max_iter = 500
solver.with_callbacks = True
solver.use_filter_line_search = True
solver.filter_size = max_iter
solver.termination_tolerance = 1e-4
solver.eps_abs = 1e-6
solver.eps_rel = 1e-6

# Warmstart the solver and solve the OCP
xs_init = [ x0 for i in range(ocp.T+1) ]
us_init = ocp.quasiStatic(xs_init[:-1])
solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)


# Plot the OCP solution using the OCP data helper class
ocp_dh   = OCPDataHandlerClassical(ocp)
ocp_data = ocp_dh.extract_data(solver.xs, solver.us) 

ocp_dh.plot_ocp_results(ocp_data, markers=['.'], SHOW=True)





# # CSSQP solver parameters
# SOLVER: 'cssqp'
