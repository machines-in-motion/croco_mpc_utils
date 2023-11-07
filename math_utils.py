"""
@package croco_mpc_utils
@file math_utils.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2023-10-18
@brief Math utils e.g. trajectory generation
"""

import numpy as np

import pinocchio as pin

import pathlib
import os
os.sys.path.insert(1, str(pathlib.Path('.').absolute()))

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

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