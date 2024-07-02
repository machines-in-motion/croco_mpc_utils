'''
This code shows how to setup quickly an Optimal Control Problem (OCP) 
from a templated YAML file (with constraints)
'''

import numpy as np

from mim_robots.robot_loader import load_pinocchio_wrapper

from croco_mpc_utils.utils import load_yaml_file
from croco_mpc_utils.ocp_constraints import OptimalControlProblemClassicalWithConstraints
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical
from croco_mpc_utils import pinocchio_utils

import mim_solvers

# Read YAML config file
config = load_yaml_file('ocp_constraint.yml')

# Import robot model (pinocchio wrapper)
robot  = load_pinocchio_wrapper('iiwa_convex')

# Setup the OCP using the wrapper
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])
ocp = OptimalControlProblemClassicalWithConstraints(robot, config).initialize(x0)

# Circle reference trajectory over the horizon
radius = 0.3 ; omega = 2.
p0 = pinocchio_utils.get_p(q0, robot, robot.model.getFrameId(config['frameTranslationFrameName']))
for i in range(ocp.T+1):
    if(i < ocp.T):
        ocp.runningModels[i].differential.costs.costs['translation'].cost.residual.reference = np.array([p0[0],
                                                                                                        p0[1] + radius * np.sin(i*config['dt']*omega), 
                                                                                                        p0[2] + radius * (1-np.cos(i*config['dt']*omega)) ])
    else:
        ocp.terminalModel.differential.costs.costs['translation'].cost.residual.reference = np.array([p0[0],
                                                                                                        p0[1] + radius * np.sin(i*config['dt']*omega), 
                                                                                                        p0[2] + radius * (1-np.cos(i*config['dt']*omega)) ])
# Initialize OCP solver
solver = mim_solvers.SolverCSQP(ocp)
solver.max_qp_iters = 1000
max_iter = 100
solver.with_callbacks = True
solver.use_filter_line_search = True
solver.filter_size = max_iter
solver.termination_tolerance = 1e-4
solver.eps_abs = 1e-6
solver.eps_rel = 1e-6

# Warmstart the solver and solve the OCP
xs_init = [ x0 for i in range(ocp.T+1) ]
us_init = ocp.quasiStatic(xs_init[:-1])
solver.solve(xs_init, us_init, maxiter=max_iter, isFeasible=False)


# Plot the OCP solution using the OCP data helper class
ocp_dh   = OCPDataHandlerClassical(ocp)
ocp_data = ocp_dh.extract_data(solver.xs, solver.us) 

fig, ax = ocp_dh.plot_ocp_results(ocp_data, markers=['.'], SHOW=True)

# ax['ee_lin'].plot()



# # CSSQP solver parameters
# SOLVER: 'cssqp'
