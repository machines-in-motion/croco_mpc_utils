"""
@package croco_mpc_utils
@file ocp_core.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2023-10-18
@brief Wrapper around Crocoddyl's API to initialize an OCP from a templated YAML config file
"""

import numpy as np

import crocoddyl
import pinocchio as pin
import hppfcl

import pathlib
import os
os.sys.path.insert(1, str(pathlib.Path('.').absolute()))

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


    
class OptimalControlProblemAbstract:
  '''
  Abstract class for Optimal Control Problem (OCP) with Crocoddyl
    robot       : pinocchio robot wrapper
    config      : dict from YAML config file of OCP params
  '''
  def __init__(self, robot, config):

    self.__dict__ = config
    
    self.rmodel = robot.model
    self.rdata = robot.data

    self.nq = robot.model.nq
    self.nv = robot.model.nv
    self.nx = self.nq + self.nv 
    
  def check_attribute(self, attribute):
    '''
    Check whether the attribute exists and is well-defined
    '''
    assert(type(attribute)==str), "Attribute to be checked must be a string"
    if(not hasattr(self, attribute)):
      logger.error("The OCP config parameter : "+str(attribute)+ " has not been defined ! Please correct the YAML config file.")
      return False
    else: return True 
     
  def check_config(self):
    self.check_attribute('SOLVER')
    self.check_attribute('dt')
    self.check_attribute('N_h')
    self.check_attribute('maxiter')
    self.check_attribute('q0')
    self.check_attribute('dq0')
    self.check_attribute('WHICH_COSTS')
    self.check_attribute('armature')

  def create_contact_model(self, contact_config, state, actuation):
    '''
    Initialize crocoddyl contact model from contact YAML Config
    '''
    # Check that contact config is complete
    self.check_attribute('contacts')
    desired_keys = ['contactModelFrameName',
                    'pinocchioReferenceFrame', 
                    'contactModelType',
                    'contactModelTranslationRef',
                    'contactModelRotationRef',
                    'contactModelGains']
    for key in desired_keys:
      if(key not in contact_config.keys()):
        logger.error("No "+key+" found in contact config !")
    # Parse arguments
    contactModelGains = np.asarray(contact_config['contactModelGains'])
    contactModelFrameName = contact_config['contactModelFrameName']
    contactModelFrameId = self.rmodel.getFrameId(contactModelFrameName)
    contactModelTranslationRef = np.asarray(contact_config['contactModelTranslationRef'])
    contactModelRotationRef = contact_config['contactModelRotationRef']
    contactModelType = contact_config['contactModelType']

    # Default reference of contact model if not specified in config
    if(contactModelTranslationRef == ''): 
      contactModelTranslationRef =  self.rdata.oMf[contactModelFrameId].translation.copy()  
    if(contactModelRotationRef == ''):
      contactModelRotationRef = self.rdata.oMf[contactModelFrameId].rotation.copy()
    
    # Detect pinocchio reference frame
    if(contact_config['pinocchioReferenceFrame'] == 'LOCAL'):
      pinocchioReferenceFrame = pin.LOCAL
    elif(contact_config['pinocchioReferenceFrame'] == 'WORLD'):
      pinocchioReferenceFrame = pin.WORLD
    elif(contact_config['pinocchioReferenceFrame'] == 'LOCAL_WORLD_ALIGNED'):
      pinocchioReferenceFrame = pin.LOCAL_WORLD_ALIGNED
    else: 
      logger.error('Unknown pinocchio reference frame. Please select0 in {LOCAL, WORLD, LOCAL_WORLD_ALIGNED} !')
    # 1D contact model = constraint an arbitrary translation (fixed position)
    if('1D' in contactModelType):
      if('x' in contactModelType):
        referenceFrameRotation = pin.utils.rpyToMatrix(0, -np.pi/2, 0)
      elif('y' in contactModelType):
        referenceFrameRotation = pin.utils.rpyToMatrix(-np.pi/2, 0, 0)
      elif('z' in contactModelType):
        referenceFrameRotation = np.eye(3)
    if('1D' in contactModelType):
      contactModel = crocoddyl.ContactModel1D(state, 
                                              contactModelFrameId, 
                                              contactModelTranslationRef,
                                              pinocchioReferenceFrame,
                                              actuation.nu,
                                              contactModelGains, 
                                              referenceFrameRotation)  
    # 3D contact model = constraint in x,y,z translations (fixed position)
    elif(contactModelType == '3D'):
      contactModel = crocoddyl.ContactModel3D(state, 
                                              contactModelFrameId, 
                                              contactModelTranslationRef,
                                              pinocchioReferenceFrame,
                                              actuation.nu, 
                                              contactModelGains, 
                                              referenceFrameRotation)  
    # 6D contact model = constraint in x,y,z translations **and** rotations (fixed placement)
    elif(contactModelType == '6D'):
      contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
      contactModel = crocoddyl.ContactModel6D(state, 
                                              contactModelFrameId, 
                                              contactModelPlacementRef, 
                                              pinocchioReferenceFrame,
                                              actuation.nu,
                                              contactModelGains, 
                                              referenceFrameRotation)     
    else: logger.error("Unknown contactModelType. Please select in {1Dx, 1Dy, 1Dz, 3D, 6D}")
    return contactModel

  def create_state_reg_cost(self, state, actuation):
    '''
    Create state regularization cost model residual 
    '''
    # Check attributes 
    self.check_attribute('stateRegRef')
    self.check_attribute('stateRegWeights')
    self.check_attribute('stateRegWeight')
    # Default reference = initial state
    if(self.stateRegRef == 'DEFAULT' or self.stateRegRef == ''):
      stateRegRef = np.concatenate([np.asarray(self.q0), np.asarray(self.dq0)]) 
    else:
      stateRegRef = np.asarray(self.stateRegRef)
    stateRegWeights = np.asarray(self.stateRegWeights)
    xRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                          crocoddyl.ResidualModelState(state, stateRegRef, actuation.nu))
    return xRegCost

  def create_ctrl_reg_cost(self, state):
    '''
    Create state regularization cost model residual 
    '''
    # Check attributes 
    self.check_attribute('ctrlRegRef')
    self.check_attribute('ctrlRegWeights')
    self.check_attribute('ctrlRegWeight')
    # Default reference 
    if(self.ctrlRegRef=='DEFAULT' or self.ctrlRegRef == ''):
      u_reg_ref = np.zeros(self.nq)
    else:
      u_reg_ref = np.asarray(self.ctrlRegRef)
    residual = crocoddyl.ResidualModelControl(state, u_reg_ref)
    ctrlRegWeights = np.asarray(self.ctrlRegWeights)
    uRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                          residual)
    return uRegCost

  def create_ctrl_reg_grav_cost(self, state):
    '''
    Create control gravity torque regularization cost model
    '''
    self.check_attribute('ctrlRegGravWeights')
    self.check_attribute('ctrlRegGravWeight')
    if(self.nb_contacts > 0):
      residual = crocoddyl.ResidualModelContactControlGrav(state)
    else:
      residual = crocoddyl.ResidualModelControlGrav(state)
    ctrlRegGravWeights = np.asarray(self.ctrlRegGravWeights)
    uRegGravCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(ctrlRegGravWeights**2), 
                                          residual)
    return uRegGravCost
  
  def create_state_limit_cost(self, state, actuation):
    '''
    Create state limit penalization cost model
    '''
    self.check_attribute('coef_xlim')
    self.check_attribute('stateLimWeights')
    self.check_attribute('stateLimWeight')
    stateLimRef = np.zeros(self.nq+self.nv)
    x_max = self.coef_xlim*state.ub 
    x_min = self.coef_xlim*state.lb
    stateLimWeights = np.asarray(self.stateLimWeights)
    xLimitCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(x_min, x_max), stateLimWeights), 
                                          crocoddyl.ResidualModelState(state, stateLimRef, actuation.nu))
    return xLimitCost

  def create_ctrl_limit_cost(self, state):
    '''
    Create control limit penalization cost model 
    '''
    self.check_attribute('coef_ulim')
    self.check_attribute('ctrlLimWeights')
    self.check_attribute('ctrlLimWeight')
    ctrlLimRef = np.zeros(self.nq)
    u_min = -self.coef_ulim*state.pinocchio.effortLimit 
    u_max = +self.coef_ulim*state.pinocchio.effortLimit 
    ctrlLimWeights = np.asarray(self.ctrlLimWeights)
    uLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max), ctrlLimWeights), 
                                            crocoddyl.ResidualModelControl(state, ctrlLimRef))
    return uLimitCost

  def create_frame_placement_cost(self, state, actuation):
    '''
    Create frame placement (SE3) cost model 
    '''
    self.check_attribute('framePlacementFrameName')
    self.check_attribute('framePlacementTranslationRef')
    self.check_attribute('framePlacementRotationRef')
    self.check_attribute('framePlacementWeights')
    self.check_attribute('framePlacementWeight')
    framePlacementFrameId = self.rmodel.getFrameId(self.framePlacementFrameName)
    # Default translation reference = initial translation
    if(self.framePlacementTranslationRef=='DEFAULT' or self.framePlacementTranslationRef==''):
      framePlacementTranslationRef = self.rdata.oMf[framePlacementFrameId].translation.copy()
    else:
      framePlacementTranslationRef = np.asarray(self.framePlacementTranslationRef)
    # Default rotation reference = initial rotation
    if(self.framePlacementRotationRef=='DEFAULT' or self.framePlacementRotationRef==''):
      framePlacementRotationRef = self.rdata.oMf[framePlacementFrameId].rotation.copy()
    else:
      framePlacementRotationRef = np.asarray(self.framePlacementRotationRef)
    framePlacementRef = pin.SE3(framePlacementRotationRef, framePlacementTranslationRef)
    framePlacementWeights = np.asarray(self.framePlacementWeights)
    framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                    crocoddyl.ResidualModelFramePlacement(state, 
                                                                                          framePlacementFrameId, 
                                                                                          framePlacementRef, 
                                                                                          actuation.nu)) 
    return framePlacementCost

  def create_frame_velocity_cost(self, state, actuation):
    '''
    Create frame velocity (tangent SE3) cost model
    '''
    self.check_attribute('frameVelocityFrameName')
    self.check_attribute('frameVelocityWeights')
    self.check_attribute('frameVelocityRef')
    self.check_attribute('frameVelocityWeight')
    frameVelocityFrameId = self.rmodel.getFrameId(self.frameVelocityFrameName)
    # Default reference = zero velocity
    if(self.frameVelocityRef=='DEFAULT' or self.frameVelocityRef==''):
      frameVelocityRef = pin.Motion( np.zeros(6) )
    else:
      frameVelocityRef = pin.Motion( np.asarray( self.frameVelocityRef ) )
    frameVelocityWeights = np.asarray(self.frameVelocityWeights)
    frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                    crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                    crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                        frameVelocityFrameId, 
                                                                                        frameVelocityRef, 
                                                                                        pin.WORLD, 
                                                                                        actuation.nu)) 
    return frameVelocityCost

  def create_frame_translation_cost(self, state, actuation):
    '''
    Create frame translation (R^3) cost model
    '''
    self.check_attribute('frameTranslationFrameName')
    self.check_attribute('frameTranslationRef')
    self.check_attribute('frameTranslationWeight')
    frameTranslationFrameId = self.rmodel.getFrameId(self.frameTranslationFrameName)
    if(self.frameTranslationRef=='DEFAULT' or self.frameTranslationRef==''):
      frameTranslationRef = self.rdata.oMf[frameTranslationFrameId].translation.copy()
    else:
      frameTranslationRef = np.asarray(self.frameTranslationRef)
      # logger.debug(str(frameTranslationRef))
    if(hasattr(self, 'frameTranslationWeights')): 
      frameTranslationWeights = np.asarray(self.frameTranslationWeights)
      frameTranslationActivation = crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2)
    elif(hasattr(self, 'alpha_quadflatlog')): 
      alpha_quadflatlog = self.alpha_quadflatlog
      frameTranslationActivation = crocoddyl.ActivationModelQuadFlatLog(3, alpha_quadflatlog)
    else:
      logger.error("Please specify either 'alpha_quadflatlog' or 'frameTranslationWeights' in config file")
    frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                    frameTranslationActivation, 
                                                    crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                            frameTranslationFrameId, 
                                                                                            frameTranslationRef, 
                                                                                            actuation.nu)) 
    return frameTranslationCost

  def create_frame_rotation_cost(self, state, actuation):
    '''
    Create frame rotation cost model
    '''    
    self.check_attribute('frameRotationFrameName')
    self.check_attribute('frameRotationRef')
    self.check_attribute('frameRotationWeights')
    self.check_attribute('frameRotationWeight')
    frameRotationFrameId = self.rmodel.getFrameId(self.frameRotationFrameName)
    # Default rotation reference = initial rotation
    if(self.frameRotationRef=='DEFAULT' or self.frameRotationRef==''):
      frameRotationRef = self.rdata.oMf[frameRotationFrameId].rotation.copy()
    else:
      frameRotationRef   = np.asarray(self.frameRotationRef)
    frameRotationWeights = np.asarray(self.frameRotationWeights)
    frameRotationCost    = crocoddyl.CostModelResidual(state, 
                                                        crocoddyl.ActivationModelWeightedQuad(frameRotationWeights**2), 
                                                        crocoddyl.ResidualModelFrameRotation(state, 
                                                                                            frameRotationFrameId, 
                                                                                            frameRotationRef, 
                                                                                            actuation.nu)) 
    return frameRotationCost

  def create_frame_force_cost(self, state, actuation):
    '''
    Create frame contact force cost model 
    '''
    self.check_attribute('frameForceFrameName')
    self.check_attribute('contacts')
    self.check_attribute('frameForceRef')
    self.check_attribute('frameForceWeights')
    self.check_attribute('frameForceWeight')
    frameForceFrameId = self.rmodel.getFrameId(self.frameForceFrameName) 
    found_ct_force_frame = False
    for ct in self.contacts:
      if(self.frameForceFrameName==ct['contactModelFrameName']):
        found_ct_force_frame = True
        ct_force_frame_type  = ct['contactModelType']
    if(not found_ct_force_frame):
      logger.error("Could not find force cost frame name in contact frame names. Make sure that the frame name of the force cost matches one of the contact frame names.")
    # 6D contact case : wrench = linear in (x,y,z) + angular in (Ox,Oy,Oz)
    if(ct_force_frame_type=='6D'):
      # Default force reference = zero force
      frameForceRef = pin.Force( np.asarray(self.frameForceRef) )
      frameForceWeights = np.asarray(self.frameForceWeights) 
      frameForceCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                  crocoddyl.ResidualModelContactForce(state, 
                                                                                      frameForceFrameId, 
                                                                                      frameForceRef, 
                                                                                      6, 
                                                                                      actuation.nu))
    # 3D contact case : linear force in (x,y,z) (LOCAL)
    if(ct_force_frame_type=='3D'):
      # Default force reference = zero force
      frameForceRef = pin.Force( np.asarray(self.frameForceRef) )
      frameForceWeights = np.asarray(self.frameForceWeights)[:3]
      frameForceCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                  crocoddyl.ResidualModelContactForce(state, 
                                                                                      frameForceFrameId, 
                                                                                      frameForceRef, 
                                                                                      3, 
                                                                                      actuation.nu))
    # 1D contact case : linear force along z (LOCAL)
    if('1D' in ct_force_frame_type):
      # Default force reference = zero force
      frameForceRef = pin.Force( np.asarray(self.frameForceRef) )
      frameForceWeights = np.asarray(self.frameForceWeights)[2:3]
      frameForceCost = crocoddyl.CostModelResidual(state, 
                                                  crocoddyl.ActivationModelWeightedQuad(frameForceWeights**2), 
                                                  crocoddyl.ResidualModelContactForce(state, 
                                                                                      frameForceFrameId, 
                                                                                      frameForceRef, 
                                                                                      1, 
                                                                                      actuation.nu))
    return frameForceCost

  def create_friction_force_cost(self, state, actuation):
    '''
    Create friction force cost model
    '''
    self.check_attribute('contacts')
    self.check_attribute('frictionConeFrameName')
    self.check_attribute('mu')
    self.check_attribute('frictionConeWeight')
    # self.check_attribute('frictioncon')
    # nsurf = cone_rotation.dot(np.matrix(np.array([0, 0, 1])).T)
    mu = self.mu
    frictionConeFrameId = self.rmodel.getFrameId(self.frictionConeFrameName)  
    # axis_
    cone_placement = self.rdata.oMf[frictionConeFrameId].copy()
    # Rotate 180° around x+ to make z become -z
    normal = cone_placement.rotation.T.dot(np.array([0.,0.,1.]))
    # cone_rotation = cone_placement.rotation.dot(pin.utils.rpyToMatrix(+np.pi, 0., 0.))
    # cone_rotation = self.rdata.oMf[frictionConeFrameId].rotation.copy() #contactModelPlacementRef.rotation
    frictionCone = crocoddyl.FrictionCone(np.eye(3), mu, 4, False) #, 0, 1000)
    frictionConeCost = crocoddyl.CostModelResidual(state,
                                                  crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(frictionCone.lb , frictionCone.ub)),
                                                  crocoddyl.ResidualModelContactFrictionCone(state, frictionConeFrameId, frictionCone, actuation.nu))
    return frictionConeCost

  def create_collision_cost(self, state, actuation):
    '''
    Create collision cost model
    '''
    self.check_attribute('collisionCostWeightTerminal')
    self.check_attribute('collisionCostWeight')
    self.check_attribute('collisionObstaclePosition')
    self.check_attribute('collisionObstacleSize')
    self.check_attribute('collisionThreshold')
    self.check_attribute('collisionFrameName')
    self.check_attribute('collisionCapsuleLength')
    self.check_attribute('collisionCapsuleRadius')

    # loader = hppfcl.MeshLoader()
    # path = "/home/skleff/devel/workspace/src/robot_properties_kuka/src/robot_properties_kuka/robot_properties_kuka/meshes/stl/iiwa_link_6.stl"
    # mesh: hppfcl.BVHModelBase = loader.load(path)
    # mesh.buildConvexHull(True, "Qt")
    # shape1 = mesh.convex

    # Create pinocchio geometry of capsule around link of interest
    collisionFrameId = self.rmodel.getFrameId(self.collisionFrameName)
    parentJointId = self.rmodel.frames[collisionFrameId].parent
    geom_model = pin.GeometryModel()
    se3_pose = self.rmodel.frames[collisionFrameId].placement
    # logger.debug(str(se3_pose))
    # se3_pose.translation = np.zeros(3) #np.array([-0,0.,0])
    ig_arm = geom_model.addGeometryObject(pin.GeometryObject("simple_arm", 
                                          collisionFrameId, 
                                          parentJointId, 
                                          hppfcl.Capsule(self.collisionCapsuleRadius, self.collisionCapsuleLength), #shape1
                                          se3_pose))
    # Add obstacle in the world
    se3_pose.translation = np.zeros(3)
    ig_obs = geom_model.addGeometryObject(pin.GeometryObject("simple_obs",
                                                             self.rmodel.getFrameId("universe"),
                                                             self.rmodel.frames[self.rmodel.getFrameId("universe")].parent,
                                                             hppfcl.Capsule(0, self.collisionObstacleSize),
                                                             se3_pose))
    # Create collision pair 
    geom_model.addCollisionPair(pin.CollisionPair(ig_arm, ig_obs))
    # Add collision cost 
    pair_id = 0
    collision_radius = 0.1 #self.collisionCapsuleRadius + self.collisionThreshold + self.collisionObstacleSize
    actCollision = crocoddyl.ActivationModel2NormBarrier(3, collision_radius)
    collisionCost = crocoddyl.CostModelResidual(state, 
                                                actCollision, 
                                                crocoddyl.ResidualModelPairCollision(state, 
                                                                                     actuation.nu, 
                                                                                     geom_model, 
                                                                                     pair_id, 
                                                                                     parentJointId)) 
    return collisionCost
    



  def create_state_constraint(self, state, actuation):
    '''
    Create state box constraint model 
    '''
    # Check attributes 
    self.check_attribute('stateLowerLimit')
    self.check_attribute('stateUpperLimit')            
    # Lower
    if(self.stateLowerLimit == 'None'):
      clip_state_min = -np.array([np.inf]*state.nx)
    elif(self.stateLowerLimit == 'DEFAULT'):
      clip_state_min = state.lb 
    else:
      clip_state_min = np.asarray(self.stateLowerLimit)
    # Upper
    if(self.stateUpperLimit == 'None'):
      clip_state_max = np.array([np.inf]*state.nx)
    elif(self.stateUpperLimit == 'DEFAULT'):
      clip_state_max = state.ub 
    else:
      clip_state_max = np.asarray(self.stateUpperLimit)
    xBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelState(state, actuation.nu), clip_state_min, clip_state_max)  
    return xBoxCstr
  
  def create_ctrl_constraint(self, state, actuation):
    '''
    Create control box constraint model 
    '''
    # Check attributes 
    self.check_attribute('ctrlLimit')
    if(self.ctrlLimit == 'None'):
      clip_ctrl = np.array([np.inf]*actuation.nu)
    elif(self.ctrlLimit == 'DEFAULT'):
      clip_ctrl = state.pinocchio.effortLimit
    else:
      clip_ctrl = np.asarray(self.ctrlLimit)
    uBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelControl(state, actuation.nu), -clip_ctrl, clip_ctrl)  
    return uBoxCstr
  
  def create_translation_constraint(self, state, actuation):
    '''
    Create end-effector position box constraint model 
    '''
    # Check attributes 
    self.check_attribute('eeLowerLimit')
    self.check_attribute('eeUpperLimit')
    self.check_attribute('eeConstraintFrameName')
    # Lower
    if(self.eeLowerLimit == 'None'):
      lmin = -np.array([np.inf]*3)
    else:
      lmin = np.asarray(self.eeLowerLimit)
    # upper
    if(self.eeUpperLimit == 'None'):
      lmax = np.array([np.inf]*3)
    else:
      lmax = np.asarray(self.eeUpperLimit)
    fid = self.rmodel.getFrameId(self.eeConstraintFrameName)
    eeBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelFrameTranslation(state, fid, np.zeros(3), actuation.nu), lmin, lmax)
    return eeBoxCstr

  def create_force_constraint(self, state, actuation):
    '''
    Create contact force box constraint model 
    '''
    # Check attributes 
    self.check_attribute('forceLowerLimit')
    self.check_attribute('forceUpperLimit')
    self.check_attribute('forceConstraintFrameName')
    self.check_attribute('forceConstraintReferenceFrame')
    self.check_attribute('forceConstraintType')
    if(self.forceConstraintType == '6D'):
      nc= 6
    elif(self.forceConstraintType == '3D'):
      nc=3
    elif('1D' in self.forceConstraintType):
      nc = 1
      mask = 2
      # if('x' in self.forceConstraintType): mask = 0
      # elif('y' in self.forceConstraintType): mask = 1
      # elif('z' in self.forceConstraintType): mask = 2
      # else: logger.error("Force constraint 1D must be in [1Dx, 1Dy, 1Dz]")
    else:
      logger.error("Force constraint type must be in [1Dx, 1Dy, 1Dz, 3D, 6D]")
    # Lower
    if(self.forceLowerLimit == 'None'):
      lmin = -np.array([np.inf]*nc)
    else:
      lmin = np.asarray(self.forceLowerLimit)
    # upper
    if(self.forceUpperLimit == 'None'):
      lmax = np.array([np.inf]*nc)
    else:
      lmax = np.asarray(self.forceUpperLimit)
    fid = self.rmodel.getFrameId(self.forceConstraintFrameName)
    if(nc==6):
      forceBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelContactForce(state, fid, pin.Force.Zero(), 6, actuation.nu), lmin, lmax)
    elif(nc==3):
      forceBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelContactForce(state, fid, pin.Force.Zero(), 3, actuation.nu), lmin[:3], lmax[:3])
    elif(nc==1):
      forceBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelContactForce(state, fid, pin.Force.Zero(), 1, actuation.nu), np.array([lmin[mask]]), np.array([lmax[mask]]))
    else:
      logger.error("Force constraint should be of type 1d, 3d or 6d !")
    return forceBoxCstr 











# Cost weights profiles, useful for reaching tasks/cost design
def cost_weight_tanh(i, N, max_weight=1., alpha=1., alpha_cut=0.25):
    '''
    Monotonically increasing weight profile over [0,...,N]
    based on a custom scaled hyperbolic tangent 
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >0)
       alpha      : controls the sharpness of the tanh (alpha high <=> very sharp)
       alpha_cut  : shifts tanh over the time window (i.e. time of inflexion point)
     OUPUT:
       Cost weight at step i : it tarts at weight=0 (when i=0) and
       ends at weight<= max_weight (at i=N). As alpha --> inf, we tend
       toward max_weight
    '''
    return 0.5*max_weight*( np.tanh(alpha*(i/N) -alpha*alpha_cut) + np.tanh(alpha*alpha_cut) )

def cost_weight_tanh_demo():
    '''
     Demo of hyperbolic tangent profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    weights_4 = np.zeros(N)
    weights_5 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_tanh(i, N, alpha=4., alpha_cut=0.50)
      weights_2[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.50)
      weights_3[i] = cost_weight_tanh(i, N, alpha=4., alpha_cut=0.75)
      weights_4[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.75)
      weights_5[i] = cost_weight_tanh(i, N, alpha=10., alpha_cut=0.25)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p0, = plt.plot(span, [1.]*N, 'k-.', label='max_weight=1')
    p1, = plt.plot(span, weights_1, 'r-', label='alpha=4, cut=0.50')
    p2, = plt.plot(span, weights_2, 'g-', label='alpha=10, cut=0.50')
    p3, = plt.plot(span, weights_3, 'b-', label='alpha=4, cut=0.75')
    p4, = plt.plot(span, weights_4, 'y-', label='alpha=10, cut=0.75')
    p5, = plt.plot(span, weights_5, 'c-', label='alpha=10, cut=0.25')
    plt.legend(handles=[p1, p2, p3, p4, p5])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_linear(i, N, min_weight=0., max_weight=1.):
    '''
    Linear cost weight profile over [0,...,N]
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : value of the weight at the end of the window (must be >=min_weight)
       min_weight : value of the weight at the start of the window (must be >=0)
     OUPUT:
       Cost weight at step i
    '''
    return (max_weight-min_weight)/N * i + min_weight

def cost_weight_linear_demo():
    '''
     Demo of linear profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_linear(i, N, min_weight=0., max_weight=1)
      weights_2[i] = cost_weight_linear(i, N, min_weight=2., max_weight=4)
      weights_3[i] = cost_weight_linear(i, N, min_weight=0, max_weight=2)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0, max_weight=1')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=2, max_weight=4')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0, max_weight=2')
    plt.legend(handles=[p1, p2, p3])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_parabolic(i, N, min_weight=0., max_weight=1.):
    '''
    Parabolic cost weight profile over [0,...,N] with min at i=N/2
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       min_weight : minimum weight reached when i=N/2
       max_weight : maximum weight reached at t=0 and i=N
     OUPUT:
       Cost weight at step i
    '''
    return min_weight + 4.*(max_weight-min_weight)/float(N**2) * (i-N/2)**2

def cost_weight_parabolic_demo():
    '''
     Demo of parabolic weight profile
    '''
    # Generate data if None provided
    N = 200
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_parabolic(i, N, min_weight=0., max_weight=4)
      weights_2[i] = cost_weight_parabolic(i, N, min_weight=2., max_weight=4)
      weights_3[i] = cost_weight_parabolic(i, N, min_weight=0, max_weight=2)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0, max_weight=4')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=2, max_weight=4')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0, max_weight=2')
    plt.legend(handles=[p1, p2, p3])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()


def cost_weight_normal_clamped(i, N, min_weight=0.1, max_weight=1., peak=1., alpha=0.01):
    '''
    Gaussian cost weight profile over [0,...,N] with max at i=N/2
     INPUT: 
       i          : current time step in the window (e.g. OCP horizon or sim horizon)
       N          : total number of time steps
       max_weight : maximum weight reached at t=N/2
     OUPUT:
       Cost weight at step i
    '''
    return min(max(peak*np.exp(-alpha*float(i-N/2)**2/2), min_weight), max_weight)

def cost_weight_normal_clamped_demo():
    '''
     Demo of clamped normal (Gaussian) weight profile
    '''
    # Generate data if None provided
    N = 500
    weights_1 = np.zeros(N)
    weights_2 = np.zeros(N)
    weights_3 = np.zeros(N)
    weights_4 = np.zeros(N)
    for i in range(N):
      weights_1[i] = cost_weight_normal_clamped(i, N, min_weight=0., max_weight=np.inf, peak=1, alpha=0.0001)
      weights_2[i] = cost_weight_normal_clamped(i, N, min_weight=0., max_weight=np.inf, peak=1, alpha=0.01)
      weights_3[i] = cost_weight_normal_clamped(i, N, min_weight=0.0, max_weight=0.8, peak=1, alpha=0.001)
      weights_4[i] = cost_weight_normal_clamped(i, N, min_weight=0.2, max_weight=1.1, peak=1.2, alpha=0.002)
    # Plot
    import matplotlib.pyplot as plt
    span = np.linspace(0, N-1, N)
    p1, = plt.plot(span, weights_1, 'r-', label='min_weight=0., max_weight=np.inf, peak=1, alpha=0.0001')
    p2, = plt.plot(span, weights_2, 'g-', label='min_weight=0., max_weight=np.inf, peak=1, alpha=0.01')
    p3, = plt.plot(span, weights_3, 'b-', label='min_weight=0.0, max_weight=0.8, peak=1, alpha=0.001')
    p4, = plt.plot(span, weights_4, 'y-', label='min_weight=0.2, max_weight=1.1, peak=1.2, alpha=0.002')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.xlabel('N')
    plt.ylabel('Cost weight')
    plt.grid()
    plt.show()



def activation_decreasing_exponential(r, alpha=1., max_weight=1., min_weight=0.5):
    '''
    Activation function of decreasing exponential clamped btw max and min 
     INPUT: 
       r          : residual 
       alpha      : sharpness of the decreasing exponential
       min_weight : minimum weight when r->infty (clamp)
       max_weight : maximum weight when r->0 (clamp)
     OUPUT:
       Cost activation
    '''
    return max(min(np.exp(1/(alpha*r+1))-1, max_weight), min_weight)




# Utils for circle trajectory tracking (position of EE frame) task

def circle_point_LOCAL_XY(t, radius=1., omega=1.):
  '''
  Returns the LOCAL frame coordinates (x,y,z) of the point reached at time t
  on a circular trajectory with given radius and angular velocity 
  The circle belongs to the LOCAL (x,y)-plane of the initial frame of interest
  starting from the top (+pi/2) and rotating clockwise
   INPUT
     t      : time (s)
     radius : radius of the circle trajectory
     omega  : angular velocity of the frame along the circle trajectory
   OUTPUT
     _      : point (x,y,z) in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  # point_LOCAL = np.array([radius*(1-np.cos(-omega*t)), radius*np.sin(-omega*t), 0.]) # (x,y)_L plane, centered in (0,-R)
  point_LOCAL = np.array([-radius*np.sin(omega*t), radius*(1-np.cos(omega*t)), 0.])    # (x,y)_L plane, centered in (0,+R)
  return point_LOCAL


def circle_point_LOCAL_XZ(t, radius=1., omega=1.):
  '''
  Returns the LOCAL frame coordinates (x,y,z) of the point reached at time t
  on a circular trajectory with given radius and angular velocity 
  The circle belongs to the LOCAL (x,z)-plane of the initial frame of interest
  starting from the top (+pi/2) and rotating clockwise
   INPUT
     t      : time (s)
     radius : radius of the circle trajectory
     omega  : angular velocity of the frame along the circle trajectory
   OUTPUT
     _      : point (x,y,z) in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  point_LOCAL = np.array([-radius*np.sin(omega*t), 0.,  radius*(1-np.cos(omega*t))])  # (x,z)_L plane, centered in (0,+R)
  return point_LOCAL


def circle_point_LOCAL_YZ(t, radius=1., omega=1.):
  '''
  Returns the LOCAL frame coordinates (x,y,z) of the point reached at time t
  on a circular trajectory with given radius and angular velocity 
  The circle belongs to the LOCAL (y,z)-plane of the initial frame of interest
  starting from the top (+pi/2) and rotating clockwise
   INPUT
     t      : time (s)
     radius : radius of the circle trajectory
     omega  : angular velocity of the frame along the circle trajectory
   OUTPUT
     _      : point (x,y,z) in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  point_LOCAL = np.array([0., -radius*np.sin(omega*t),  radius*(1-np.cos(omega*t))])  # (y,z)_L plane, centered in (0,+R)
  return point_LOCAL


def circle_point_WORLD(t, M, radius=1., omega=1., LOCAL_PLANE='XY'):
  '''
  Returns the WORLD frame coordinates (x,y,z) of the point reached at time t
  on a circular trajectory with given radius and angular velocity 
   INPUT
     t           : time (s)
     M           : initial placement of the frame of interest (pinocchio.SE3)   
     radius      : radius of the circle trajectory
     omega       : angular velocity of the frame along the circle trajectory
     LOCAL_PLANE : in which plane of the LOCAL frame lies the circle {'XY', 'XZ', 'YZ'}
   OUTPUT
     _      : point (x,y,z) in WORLD frame (np.array)
  '''
  # WORLD coordinates 
  if(LOCAL_PLANE=='XY'):
    point_WORLD = M.act(circle_point_LOCAL_XY(t, radius=radius, omega=omega))
  elif(LOCAL_PLANE=='XZ'):
    point_WORLD = M.act(circle_point_LOCAL_XZ(t, radius=radius, omega=omega))
  elif(LOCAL_PLANE=='YZ'):
    point_WORLD = M.act(circle_point_LOCAL_YZ(t, radius=radius, omega=omega))
  else:
    logger.error("Unknown LOCAL_PLANE for circle trajectory. Choose LOCAL_PLANE in {'XY', 'XZ', 'YZ'}")
  return point_WORLD




# Utils for rotation trajectory tracking (orientation of EE frame) task

def rotation_orientation_LOCAL_X(t, omega=1.):
  '''
  Returns the orientation matrix w.r.t. LOCAL frame reached at time t
  when rotating about the LOCAL x-axis at constant angular velocity
   INPUT
     t      : time (s)
     omega  : angular velocity of the frame rotating about x-LOCAL (w.r.t. LOCAL)
   OUTPUT
     _      : orientation 3x3 matrix in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  rotation_LOCAL = pin.utils.rpyToMatrix(np.array([np.sin(omega*t), 0., 0.]))
  return rotation_LOCAL


def rotation_orientation_LOCAL_Y(t, omega=1.):
  '''
  Returns the orientation matrix w.r.t. LOCAL frame reached at time t
  when rotating about the LOCAL y-axis at constant angular velocity
   INPUT
     t      : time (s)
     omega  : angular velocity of the frame rotating about y-LOCAL (w.r.t. LOCAL)
   OUTPUT
     _      : orientation 3x3 matrix in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  rotation_LOCAL = pin.utils.rpyToMatrix(np.array([0., np.sin(omega*t), 0.]))
  return rotation_LOCAL


def rotation_orientation_LOCAL_Z(t, omega=1.):
  '''
  Returns the orientation matrix w.r.t. LOCAL frame reached at time t
  when rotating about the LOCAL z-axis at constant angular velocity
   INPUT
     t      : time (s)
     omega  : angular velocity of the frame rotating about z-LOCAL (w.r.t. LOCAL)
   OUTPUT
     _      : orientation 3x3 matrix in LOCAL frame (np.array)
  '''
  # LOCAL coordinates 
  rotation_LOCAL = pin.utils.rpyToMatrix(np.array([0., 0., np.sin(omega*t)]))
  return rotation_LOCAL


def rotation_orientation_WORLD(t, M, omega=1., LOCAL_AXIS='Z'):
  '''
  Returns the WORLD frame coordinates (x,y,z) of the point reached at time t
  on a circular trajectory with given radius and angular velocity 
   INPUT
     t           : time (s)
     M           : initial placement of the frame of interest (pinocchio.SE3)   
     radius      : radius of the circle trajectory
     omega       : angular velocity of the frame along the circle trajectory
     LOCAL_AXIS  : LOCAL axis about which the LOCAL frame rotates {'X', 'Y', 'Z'}
   OUTPUT
     _      : orientation 3x3 matrix in WORLD frame (np.array)
  '''
  # WORLD coordinates 
  if(LOCAL_AXIS=='X'):
    orientation_WORLD = M.rotation.copy().dot(rotation_orientation_LOCAL_X(t, omega=omega))
  elif(LOCAL_AXIS=='Y'):
    orientation_WORLD = M.rotation.copy().dot(rotation_orientation_LOCAL_Y(t, omega=omega))
  elif(LOCAL_AXIS=='Z'):
    orientation_WORLD = M.rotation.copy().dot(rotation_orientation_LOCAL_Z(t, omega=omega))
  else:
    logger.error("Unknown LOCAL_AXIS for circle trajectory. Choose LOCAL_AXIS in {'X', 'Y', 'Z'}")
  return orientation_WORLD