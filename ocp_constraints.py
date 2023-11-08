
"""
@package croco_mpc_utils
@file ocp_constraints.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2023-10-18
@brief Wrapper around Crocoddyl's API to initialize an OCP from a templated YAML config file (with constraints)
"""


import crocoddyl

from croco_mpc_utils.ocp import OptimalControlProblemClassical

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


class OptimalControlProblemClassicalWithConstraints(OptimalControlProblemClassical):
  '''
  Helper class for constrained OCP setup with Crocoddyl
  '''
  def __init__(self, robot, config):
    '''
    Override base class constructor if necessary
    '''
    super().__init__(robot, config)
  
  def check_config(self):
    '''
    Override base class checks if necessary
    '''
    super().check_config()
    self.check_attribute('WHICH_CONSTRAINTS')
    self.check_attribute('use_filter_ls')
    self.check_attribute('filter_size')
    self.check_attribute('warm_start')
    self.check_attribute('termination_tol')
    self.check_attribute('max_qp_iters')
    self.check_attribute('qp_termination_tol_abs')
    self.check_attribute('qp_termination_tol_rel')
    self.check_attribute('warm_start_y')
    self.check_attribute('reset_rho')

  def create_constraint_model_manager(self, state, actuation, node_id):
    '''
    Initialize a constraint model manager and adds constraints to it 
    '''
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      stateBoxConstraint = self.create_state_constraint(state, actuation)   
      constraintModelManager.addConstraint('stateBox', stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      constraintModelManager.addConstraint('ctrlBox', ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      translationBoxConstraint = self.create_translation_constraint(state, actuation)
      constraintModelManager.addConstraint('translationBox', translationBoxConstraint)
    # Contact force 
    if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0 and node_id != self.N_h):
      forceBoxConstraint = self.create_force_constraint(state, actuation)
      constraintModelManager.addConstraint('forceBox', forceBoxConstraint)

  def create_differential_action_model(self, state, actuation, constraintModelManager):
    '''
    Initialize a differential action model with or without contacts, 
    and with explicit constraints
    '''
    # If there are contacts, defined constrained DAM
    contactModels = []
    if(self.nb_contacts > 0):
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))   
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  constraintModelManager,
                                                                  inv_damping=0., 
                                                                  enable_force=True)
    # Otherwise just create free DAM
    else:
      dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                              actuation, 
                                                              crocoddyl.CostModelSum(state, nu=actuation.nu),
                                                              constraintModelManager)
    return dam, contactModels

  def success_log(self):
    '''
    Log of successful OCP initialization + important information
    '''
    super().success_log()
    logger.info("    CONSTRAINTS   = "+str(self.WHICH_CONSTRAINTS))
       
  def initialize(self, x0):
    '''
    Initializes an Optimal Control Problem (OCP) from YAML config parameters and initial state
      INPUT: 
          x0          : initial state of shooting problem
      OUTPUT:
          crocoddyl.ShootingProblem object 

     A cost term on a variable z(x,u) has the generic form w * a( r( z(x,u) - z0 ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference z0, e.g. 'stateRegRef'
                  When ref is set to 'DEFAULT' in YAML file, default references hard-coded here are used
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    ''' 
  # State and actuation models
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
  
  # Contact or not ?
    self.parse_contacts()

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create constraint manager and constraints
        constraintModelManager = self.create_constraint_model_manager(state, actuation, i)
      # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts+constraints
        dam, contactModels = self.create_differential_action_model(state, actuation, constraintModelManager) 
        runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=self.dt))
        self.init_running_model(state, actuation, runningModels[i], contactModels)

    # Terminal model
    constraintModelManager = self.create_constraint_model_manager(state, actuation, self.N_h)
    dam_t, contactModels = self.create_differential_action_model(state, actuation, constraintModelManager)  
    terminalModel = crocoddyl.IntegratedActionModelEuler( dam_t, stepTime=0. )
    self.init_terminal_model(state, actuation, terminalModel, contactModels)
    
    logger.info("Created IAMs.")  


  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

  # Finish
    self.success_log()
    
    return problem


    
