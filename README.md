# croco_mpc_utils
Utilities for easy &amp; modular MPC prototyping using Crocoddyl. This package's main features are :
- wrappers around Crocoddyl's API to quickly design Optimal Control Problems (OCPs) through templated configuration files
- data handlers and plotters to visualize OCPs solutions and MPC simulation results

# Dependencies
- [crocoddyl](https://github.com/loco-3d/crocoddyl) (>=2.0)
- [pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [eigenpy](https://github.com/stack-of-tasks/eigenpy)

Standard python dependencies: matplotlib, numpy, logger, yaml, os, pathlib, setuptools

Optional dependencies (examples) : mim_robots, mim_solvers

# How to install
```
git clone https://github.com/machines-in-motion/croco_mpc_utils.git
cd croco_mpc_utils && pip install .`
```

# How to use 
Check out the examples to see how this package must be used.

The main idead is that the YAML file contains all the information required to design an OCP using Crocoddyl's API. The OCP wrappers parse the YAML file and return a `crocoddyl.ShootingProblem` object. A minimal configuration file is : 
```
dt: 0.02                                 # OCP integration step 
N_h: 100                                 # Horizon length in nodes
maxiter: 100                             # Max number of iterations in DDP
q0: [0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]   # Initial robot joint configuration
dq0: [0.,0.,0.,0.,0.,0.,0.]              # Initial robot joint velocity
WHICH_COSTS: ['']                        # Cost function terms
```
The field `WHICH_COSTS` contains strings of cost names. Supported cost names are :
- 'stateReg'     : `crocoddyl.ResidualModelState` with weighted quadratic activation
- 'ctrlReg'      : `crocoddyl.ResidualModelControl` with weighted quadratic activation
- 'ctrlRegGrav'  : `crocoddyl.ResidualModelControl` with weighted quadratic activation
- 'translation'  : `crocoddyl.ResidualModelFrameTranslation` with weighted quadratic activation
- 'velocity'     : `crocoddyl.ResidualModelFrameVelocity` with weighted quadratic activation
- 'rotation'     : `crocoddyl.ResidualModelRotation` with weighted quadratic activation
- 'placement'    : `crocoddyl.ResidualModelFramePlacement` with weighted quadratic activation
- 'force'        : `crocoddyl.ResidualModelContactForce` with weighted quadratic activation
- 'stateLim'     : `crocoddyl.ResidualModelState` with weighted quadratic barrier activation
- 'ctrlLim'      : `crocoddyl.ResidualModelControl` with weighted quadratic barrier activation
- 'friction'     : `crocoddyl.ResidualModelContactFrictionCone` with weighted quadratic barrier activation

Note that you can implement your own wrapper class inheriting from the abstract (`core_ocp.OptimalControlProblemAbstract`) or derived (`ocp.OptimalControlProblemClassical`) wrappers to fit your needs. The same goes for the data handler classes.
