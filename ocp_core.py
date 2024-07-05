"""
@package croco_mpc_utils
@file ocp_core.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2023-10-18
@brief Abstract wrapper around Crocoddyl's API to initialize an OCP from a templated YAML config file
"""

from typing import List

import numpy as np

import crocoddyl
import pinocchio as pin
import hppfcl

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


    
class OptimalControlProblemAbstract:
  '''
  Abstract class for Optimal Control Problem (OCP) with Crocoddyl
    model       : pinocchio model
    config      : dict from YAML config file of OCP params
  '''
  def __init__(self, robot, config):

    self.__dict__ = config
    
    self.check_config()

    self.rmodel = robot.model
    self.rdata = robot.data
    self.cmodel = robot.collision_model
    self.cdata = self.cmodel.createData() #! Is there a collision data in robot? 

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
    self.check_attribute('dt')
    self.check_attribute('N_h')
    self.check_attribute('maxiter')
    self.check_attribute('q0')
    self.check_attribute('dq0')
    self.check_attribute('WHICH_COSTS')

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
      contactModel = crocoddyl.ContactModel1D(state, 
                                              contactModelFrameId, 
                                              contactModelTranslationRef[2],
                                              pinocchioReferenceFrame,
                                              referenceFrameRotation,
                                              actuation.nu,
                                              contactModelGains)  
    # 3D contact model = constraint in x,y,z translations (fixed position)
    elif(contactModelType == '3D'):
      contactModel = crocoddyl.ContactModel3D(state, 
                                              contactModelFrameId, 
                                              contactModelTranslationRef,
                                              pinocchioReferenceFrame,
                                              actuation.nu, 
                                              contactModelGains)  
    # 6D contact model = constraint in x,y,z translations **and** rotations (fixed placement)
    elif(contactModelType == '6D'):
      contactModelPlacementRef = pin.SE3(contactModelRotationRef, contactModelTranslationRef)
      contactModel = crocoddyl.ContactModel6D(state, 
                                              contactModelFrameId, 
                                              contactModelPlacementRef, 
                                              pinocchioReferenceFrame,
                                              actuation.nu,
                                              contactModelGains)     
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
  
  def create_collision_constraints(self, state, actuation) -> List:
    """Create collision box constraints models. 

    Returns:
        List: List of collision constraint models.
    """
    from colmpc import ResidualDistanceCollision
    
    self.check_attribute("safetyMargin")
    self.check_attribute("collisionPairs")

    if (self.safetyMargin == "None"):
      safety = 0
    else:
      safety = float(self.safetyMargin)
    
    if (len(self.collisionPairs) == 0):
      logger.info("There is no collision pairs!")
    
    # Going through all the collisions
    for collision in self.collisionPairs:
      col1,col2 = collision[0], collision[1]
      if not self.cmodel.existGeometryName(col1) or  not self.cmodel.existGeometryName(col2):
        logger.error(f"The geometry names {col1} and/or {col2} do not exist in the current collision model.")

      # Creating the collision pair
      self.cmodel.addCollisionPair(
        pin.CollisionPair(
           self.cmodel.getGeometryId(col1),
           self.cmodel.getGeometryId(col2)
        )
      )
    
    # Now that the collision pairs are all added, creating the constraints
    
    CollisionBoxCstrs = []
    
    if len(self.cmodel.collisionPairs) != 0:
            for col_idx in range(len(self.cmodel.collisionPairs)):
                obstacleDistanceResidual = ResidualDistanceCollision(state, actuation.nu, self.cmodel, col_idx)

                # Creating the inequality constraint
                collision_constraint = crocoddyl.ConstraintModelResidual(
                    state,
                    obstacleDistanceResidual,
                    np.array([safety]),
                    np.array([np.inf]),
                )
                CollisionBoxCstrs.append(collision_constraint)
    
    return CollisionBoxCstrs