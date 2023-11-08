"""
@package croco_mpc_utils
@file ocp_core_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2023-10-18
@brief Data handlers for abstract OCP wrapper class
"""

import time
import os

import numpy as np
import pinocchio as pin

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from croco_mpc_utils import pinocchio_utils as pin_utils


# Save simulation data (dictionary) into compressed *.npz
def save_data(sim_data, save_dir, save_name=None):
    '''
    Saves data to a compressed npz file (binary)
    Args:
      sim_data  : object
      save_dir  : save directory path
      save_name : name of npz file 
    '''
    logger.info('Compressing & saving data...')
    if(save_name is None):
        save_name = 'sim_data_NO_NAME'+str(time.time())
    save_path = save_dir+'/'+save_name+'.npz'
    np.savez_compressed(save_path, data=sim_data)
    logger.info("Saved data to "+str(save_path)+" !")


# Loads simulation data dictionary from compressed *.npz
def load_data(npz_file):
    '''
    Loads a npz archive of sim_data into a dict
    Args:
      npz_file : path to *.npz
    '''
    logger.info('Loading data...')
    d = np.load(npz_file, allow_pickle=True)
    return d['data'][()]



# Abstract OCP data handler : extract data + generate fancy plots
class OCPDataHandlerAbstract:
  '''
  Abstract helper class to plot Crocoddyl OCP results 
  Plotting functions of results, state and control are purely virtual
  '''
  def __init__(self, ocp):
    '''
    Args: 
      ocp : crocoddyl.ShootingProblem 
    '''
    self.ocp = ocp
  
  def find_cost_ids(self):
    '''
    Scans all residual models of every OCP nodes to find frame ids
    '''
    frameIds = []
    for m in self.ocp.runningModels:
      for cost_name in m.differential.costs.active_set:
        residualModel = m.differential.costs.costs[cost_name].cost.residual
        if(hasattr(residualModel, 'id')):
            if(residualModel.id not in frameIds):
              frameIds.append(residualModel.id)
    logger.warning("Found "+str(len(frameIds))+" cost frame ids in the cost function !")
    # TODO: enable multi-frames
    if(len(frameIds) > 1):
       logger.warning("Found "+str(len(frameIds))+" cost frame ids in the cost function !")
       logger.warning(str(frameIds))
       logger.warning("Only the first frame id "+str(frameIds[0])+" will be plotted")
    if(len(frameIds) == 0):
       logger.warning("Did not find any cost frame id.")
    return frameIds[0]

  def find_contact_ids(self):
    '''
    Scans all residual models of every OCP nodes to find contact frame names
    '''
    frameIds = []
    for m in self.ocp.runningModels:
      if(hasattr(m.differential, 'contacts')):
        for ct_name in m.differential.contacts.contacts.todict().keys():
          ct = m.differential.contacts.contacts[ct_name].contact
          if(ct.id not in frameIds):
              frameIds.append(ct.id)
    logger.warning("Found "+str(len(frameIds))+" contact frame ids in the cost function !")
    # TODO: enable multi-frames
    if(len(frameIds) > 1):
       logger.warning("Found "+str(len(frameIds))+" contact frame ids in the cost function !")
       logger.warning(str(frameIds))
       logger.warning("Only the first frame id "+str(frameIds[0])+" will be plotted")
    if(len(frameIds) == 0):
       logger.warning("Did not find any contact frame id.")
    return frameIds[0]

  def find_contact_names(self):
    '''
    Scans all residual models of every OCP nodes to find contact frame names
    '''
    ctNames = []
    for m in self.ocp.runningModels:
      if(hasattr(m.differential, 'contacts')):
        for ct_name in m.differential.contacts.contacts.todict().keys():
          if(ct_name not in ctNames):
              ctNames.append(ct_name)
    logger.warning("Found "+str(len(ctNames))+" contact names in the cost function !")
    # TODO: enable multi-frames
    if(len(ctNames) > 1):
       logger.warning("Found "+str(len(ctNames))+" contact names in the cost function !")
       logger.warning(str(ctNames))
       logger.warning("Only the first contact name "+str(ctNames[0])+" will be plotted")
    if(len(ctNames) == 0):
       logger.warning("Did not find any contact name id.")
    return ctNames[0]
  
  # Data extraction : solution + contact + cost references
  def extract_data(self, xs, us):
    '''
    Extract relevant plotting data from (X,U) solution of the OCP
    Args:
      xs : nd array
      us : nd array
    '''
    logger.info("Extract OCP data...")
    # Store data
    ocp_data = {}
    # OCP params
    ocp_data['T'] = self.ocp.T
    ocp_data['dt'] = self.ocp.runningModels[0].dt
    ocp_data['nq'] = self.ocp.runningModels[0].state.nq
    ocp_data['nv'] = self.ocp.runningModels[0].state.nv
    ocp_data['nu'] = self.ocp.runningModels[0].differential.actuation.nu
    ocp_data['nx'] = self.ocp.runningModels[0].state.nx
    ocp_data['dts'] = [self.ocp.runningModels[i].dt for i in range(ocp_data['T'])]
    ocp_data['dts'].append(self.ocp.terminalModel.dt)
    # Pin model
    ocp_data['pin_model'] = self.ocp.runningModels[0].differential.pinocchio
    # Look for the frames ids used in the cost function residual models
    ocp_data['cost_frame_id']    = self.find_cost_ids()
    # Solution trajectories
    ocp_data['xs'] = xs
    ocp_data['us'] = us
    ocp_data['CONTACT_TYPE'] = None
    PIN_REF_FRAME =   pin.LOCAL
    # Extract force at EE frame and contact info
    if(hasattr(self.ocp.runningModels[0].differential, 'contacts')):
      ocp_data['contact_frame_id'] = self.find_contact_ids()
      ct_frame_name                = self.find_contact_names()
      # Get refs for contact model
      contactModelRef0 = self.ocp.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.reference
      # Case 6D contact (x,y,z,Ox,Oy,Oz)
      if(hasattr(contactModelRef0, 'rotation')):
        ocp_data['contact_rotation'] = [self.ocp.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.rotation for i in range(self.ocp.T)]
        ocp_data['contact_rotation'].append(self.ocp.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.rotation)
        ocp_data['contact_translation'] = [self.ocp.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference.translation for i in range(self.ocp.T)]
        ocp_data['contact_translation'].append(self.ocp.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference.translation)
        ocp_data['CONTACT_TYPE'] = '6D'
        ocp_data['nc'] = 6
        PIN_REF_FRAME =   pin.LOCAL
      # Case 3D contact (x,y,z)
      elif(np.size(contactModelRef0)==3):
        if(self.ocp.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3):
          # Get ref translation for 3D 
          ocp_data['contact_translation'] = [self.ocp.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(self.ocp.T)]
          ocp_data['contact_translation'].append(self.ocp.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
          ocp_data['CONTACT_TYPE'] = '3D'
          ocp_data['nc'] = 3
        elif(self.ocp.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 1):
          # Case 1D contact
          ocp_data['contact_translation'] = [self.ocp.runningModels[i].differential.contacts.contacts[ct_frame_name].contact.reference for i in range(self.ocp.T)]
          ocp_data['contact_translation'].append(self.ocp.terminalModel.differential.contacts.contacts[ct_frame_name].contact.reference)
          ocp_data['CONTACT_TYPE'] = '1D'
          ocp_data['nc'] = 1
        else: 
          print(self.ocp.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.nc == 3)
          logger.error("Contact must be 1D or 3D !")
        # Check which reference frame is used 
        if(self.ocp.runningModels[0].differential.contacts.contacts[ct_frame_name].contact.type == pin.pinocchio_pywrap.ReferenceFrame.LOCAL):
          PIN_REF_FRAME = pin.LOCAL
        else:
          PIN_REF_FRAME = pin.LOCAL_WORLD_ALIGNED
      # Get contact force
      datas = [self.ocp.runningDatas[i].differential.multibody.contacts.contacts[ct_frame_name] for i in range(self.ocp.T)]
      # data.f = force exerted at parent joint expressed in WORLD frame (oMi)
      # express it in LOCAL contact frame using jMf 
      ee_forces = [data.jMf.actInv(data.f).vector for data in datas] 
      ocp_data['fs'] = [ee_forces[i] for i in range(self.ocp.T)]
      # Express in WORLD aligned frame otherwise
      if(PIN_REF_FRAME == pin.LOCAL_WORLD_ALIGNED or PIN_REF_FRAME == pin.WORLD):
        # ct_frame_id = ocp_data['pin_model'].getFrameId(ct_frame_name)
        Ms = [pin_utils.get_SE3_(ocp_data['xs'][i][:ocp_data['nq']], ocp_data['pin_model'], ocp_data['contact_frame_id']) for i in range(self.ocp.T)]
        ocp_data['fs'] = [Ms[i].action @ ee_forces[i] for i in range(self.ocp.T)]
    # Extract refs for active costs 
    # TODO : active costs may change along horizon : how to deal with that when plotting? 
    ocp_data['active_costs'] = self.ocp.runningModels[0].differential.costs.active_set
    if('stateReg' in ocp_data['active_costs']):
        ocp_data['stateReg_ref'] = [self.ocp.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(self.ocp.T)]
        ocp_data['stateReg_ref'].append(self.ocp.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    if('ctrlReg' in ocp_data['active_costs']):
        ocp_data['ctrlReg_ref'] = [self.ocp.runningModels[i].differential.costs.costs['ctrlReg'].cost.residual.reference for i in range(self.ocp.T)]
    if('ctrlRegGrav' in ocp_data['active_costs']):
        ocp_data['ctrlRegGrav_ref'] = [pin_utils.get_u_grav(ocp_data['xs'][i][:ocp_data['nq']], ocp_data['pin_model']) for i in range(self.ocp.T)]
    if('stateLim' in ocp_data['active_costs']):
        ocp_data['stateLim_ub'] = [self.ocp.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.ub for i in range(self.ocp.T)]
        ocp_data['stateLim_lb'] = [self.ocp.runningModels[i].differential.costs.costs['stateLim'].cost.activation.bounds.lb for i in range(self.ocp.T)]
        ocp_data['stateLim_ub'].append(self.ocp.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.ub)
        ocp_data['stateLim_lb'].append(self.ocp.terminalModel.differential.costs.costs['stateLim'].cost.activation.bounds.lb)
    if('ctrlLim' in ocp_data['active_costs']):
        ocp_data['ctrlLim_ub'] = [self.ocp.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub for i in range(self.ocp.T)]
        ocp_data['ctrlLim_lb'] = [self.ocp.runningModels[i].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb for i in range(self.ocp.T)]
        ocp_data['ctrlLim_ub'].append(self.ocp.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.ub)
        ocp_data['ctrlLim_lb'].append(self.ocp.runningModels[-1].differential.costs.costs['ctrlLim'].cost.activation.bounds.lb)
    if('placement' in ocp_data['active_costs']):
        ocp_data['translation_ref'] = [self.ocp.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.translation for i in range(self.ocp.T)]
        ocp_data['translation_ref'].append(self.ocp.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation)
        ocp_data['rotation_ref'] = [self.ocp.runningModels[i].differential.costs.costs['placement'].cost.residual.reference.rotation for i in range(self.ocp.T)]
        ocp_data['rotation_ref'].append(self.ocp.terminalModel.differential.costs.costs['placement'].cost.residual.reference.rotation)
    if('translation' in ocp_data['active_costs']):
        ocp_data['translation_ref'] = [self.ocp.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(self.ocp.T)]
        ocp_data['translation_ref'].append(self.ocp.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    if('velocity' in ocp_data['active_costs']):
        ocp_data['velocity_ref'] = [self.ocp.runningModels[i].differential.costs.costs['velocity'].cost.residual.reference.vector for i in range(self.ocp.T)]
        ocp_data['velocity_ref'].append(self.ocp.terminalModel.differential.costs.costs['velocity'].cost.residual.reference.vector)
    if('rotation' in ocp_data['active_costs']):
        ocp_data['rotation_ref'] = [self.ocp.runningModels[i].differential.costs.costs['rotation'].cost.residual.reference for i in range(self.ocp.T)]
        ocp_data['rotation_ref'].append(self.ocp.terminalModel.differential.costs.costs['rotation'].cost.residual.reference)
    if('force' in ocp_data['active_costs']): 
        ocp_data['force_ref'] = [self.ocp.runningModels[i].differential.costs.costs['force'].cost.residual.reference.vector for i in range(self.ocp.T)]
    return ocp_data
  
  # Virtual plotting functions
  def plot_ocp_results(self):
      raise NotImplementedError()

  def plot_ocp_state(self):
      raise NotImplementedError()

  def plot_ocp_control(self):
      raise NotImplementedError()

  # Base plotting functions
  def plot_ocp_endeff_linear(self, ocp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                      MAKE_LEGEND=False, SHOW=True, AUTOSCALE=False):
      '''
      Plot OCP results (end-effector linear position, velocity)
      '''
      # Parameters
      N = ocp_data['T'] 
      dt = ocp_data['dt']
      nq = ocp_data['nq']
      nv = ocp_data['nv'] 
      # Extract EE traj
      x = np.array(ocp_data['xs'])
      q = x[:,:nq]
      v = x[:,nq:nq+nv]
      lin_pos_ee = pin_utils.get_p_(q, ocp_data['pin_model'], ocp_data['cost_frame_id'])
      lin_vel_ee = pin_utils.get_v_(q, v, ocp_data['pin_model'], ocp_data['cost_frame_id'])
      # Cost reference frame translation if any, or initial one
      if('translation' in ocp_data['active_costs'] or 'placement' in ocp_data['active_costs']):
          lin_pos_ee_ref = np.array(ocp_data['translation_ref'])
      else:
          lin_pos_ee_ref = np.array([lin_pos_ee[0,:] for i in range(N+1)])
      # Cost reference frame linear velocity if any, or initial one
      if('velocity' in ocp_data['active_costs']):
          lin_vel_ee_ref = np.array(ocp_data['velocity_ref'])[:,:3] # linear part
      else:
          lin_vel_ee_ref = np.array([lin_vel_ee[0,:] for i in range(N+1)])
      # Contact reference translation if CONTACT
      if(ocp_data['CONTACT_TYPE'] is not None):
          lin_pos_ee_contact = np.array(ocp_data['contact_translation'])
      # Plots
      tspan = np.array([sum(ocp_data['dts'][:i]) for i in range(len(ocp_data['dts']))]) #np.linspace(0, N*sum(ocp_data['dts']), N+1)
      if(ax is None or fig is None):
          fig, ax = plt.subplots(3, 2, sharex='col')
      if(label is None):
          label='OCP solution'
      xyz = ['x', 'y', 'z']
      for i in range(3):
          # Plot EE position in WORLD frame
          ax[i,0].plot(tspan, lin_pos_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
          # Plot EE target frame translation in WORLD frame
          if('translation' or 'placement' in ocp_data['active_costs']):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,0].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,0].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
          # Plot CONTACT reference frame translation in WORLD frame
          if(ocp_data['CONTACT_TYPE'] is not None):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('Baumgarte stab. ref.' in labels):
                  handles.pop(labels.index('Baumgarte stab. ref.'))
                  ax[i,0].lines.pop(labels.index('Baumgarte stab. ref.'))
                  labels.remove('Baumgarte stab. ref.')
              ax[i,0].plot(tspan, lin_pos_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)
          # Labels, tick labels, grid
          ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,0].grid(True)

          # Plot EE (linear) velocities in WORLD frame
          ax[i,1].plot(tspan, lin_vel_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)
          # Plot EE target frame (linear) velocity in WORLD frame
          if('velocity' in ocp_data['active_costs']):
              handles, labels = ax[i,1].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,1].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,1].plot(tspan, lin_vel_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
          # Labels, tick labels, grid
          ax[i,1].set_ylabel('$V^{EE}_%s$ (m/s)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,1].grid(True)
      
      #x-label + align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)

      # Set ylim if any
      if(AUTOSCALE):
          TOL = 0.1
          ax_p_ylim = 1  #1.1*max(np.max(np.abs(lin_pos_ee)), TOL)
          ax_v_ylim = 1 #1.1*max(np.max(np.abs(lin_vel_ee)), TOL)
          for i in range(3):
              ax[i,0].set_ylim(lin_pos_ee_ref[0,i]-ax_p_ylim, lin_pos_ee_ref[0,i]+ax_p_ylim) 
              ax[i,1].set_ylim(lin_vel_ee_ref[0,i]-ax_v_ylim, lin_vel_ee_ref[0,i]+ax_v_ylim)

      if(MAKE_LEGEND):
          handles, labels = ax[2,0].get_legend_handles_labels()
          fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.suptitle('End-effector frame position and linear velocity', size=18)
      if(SHOW):
          plt.show()
      return fig, ax

  def plot_ocp_endeff_angular(self, ocp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                      MAKE_LEGEND=False, SHOW=True, AUTOSCALE=False):
      '''
      Plot OCP results (endeff angular position, velocity)
      '''
      # Parameters
      N = ocp_data['T'] 
      dt = ocp_data['dt']
      nq = ocp_data['nq']
      nv = ocp_data['nv'] 
      # Extract EE traj
      x = np.array(ocp_data['xs'])
      q = x[:,:nq]
      v = x[:,nq:nq+nv]
      rpy_ee = pin_utils.get_rpy_(q, ocp_data['pin_model'], ocp_data['cost_frame_id'])
      w_ee   = pin_utils.get_w_(q, v, ocp_data['pin_model'], ocp_data['cost_frame_id'])
      # Cost reference frame orientation if any, or initial one
      if('rotation' in ocp_data['active_costs'] or 'placement' in ocp_data['active_costs']):
          rpy_ee_ref = np.array([pin.utils.matrixToRpy(np.array(R)) for R in ocp_data['rotation_ref']])
      else:
          rpy_ee_ref = np.array([rpy_ee[0,:] for i in range(N+1)])
      # Cost reference angular velocity if any, or initial one
      if('velocity' in ocp_data['active_costs']):
          w_ee_ref = np.array(ocp_data['velocity_ref'])[:,3:] # angular part
      else:
          w_ee_ref = np.array([w_ee[0,:] for i in range(N+1)])
      # Contact reference orientation (6D)
      if(ocp_data['CONTACT_TYPE']=='6D'):
          rpy_ee_contact = np.array([pin.utils.matrixToRpy(R) for R in ocp_data['contact_rotation']])
      # Plots
      tspan = np.array([sum(ocp_data['dts'][:i]) for i in range(len(ocp_data['dts']))]) #np.linspace(0, N*sum(ocp_data['dts']), N+1)
      if(ax is None or fig is None):
          fig, ax = plt.subplots(3, 2, sharex='col')
      if(label is None):
          label='OCP solution'
      xyz = ['x', 'y', 'z']
      for i in range(3):
          # Plot EE orientation in WORLD frame
          ax[i,0].plot(tspan, rpy_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot EE target frame orientation in WORLD frame
          if('rotation' or 'placement' in ocp_data['active_costs']):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,0].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,0].plot(tspan, rpy_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
          
          # Plot CONTACT reference frame rotation in WORLD frame
          if(ocp_data['CONTACT_TYPE']=='6D'):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('contact' in labels):
                  handles.pop(labels.index('contact'))
                  ax[i,0].lines.pop(labels.index('contact'))
                  labels.remove('contact')
              ax[i,0].plot(tspan, rpy_ee_contact[:,i], linestyle=':', color='r', marker=None, label='Baumgarte stab. ref.', alpha=0.3)

          # Labels, tick labels, grid
          ax[i,0].set_ylabel('$RPY^{EE}_%s$ (rad)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,0].grid(True)

          # Plot EE 'linear) velocities in WORLD frame
          ax[i,1].plot(tspan, w_ee[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot EE target frame (linear) velocity in WORLD frame
          if('velocity' in ocp_data['active_costs']):
              handles, labels = ax[i,1].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,1].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,1].plot(tspan, w_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)
          
          # Labels, tick labels, grid
          ax[i,1].set_ylabel('$W^{EE}_%s$ (rad/s)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,1].grid(True)
      
      #x-label + align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)

      # Set ylim if any
      if(AUTOSCALE):
          TOL = 0.1
          ax_p_ylim = 1.1*max(np.max(np.abs(rpy_ee)), TOL)
          ax_v_ylim = 1.1*max(np.max(np.abs(w_ee)), TOL)
          for i in range(3):
              ax[i,0].set_ylim(-ax_p_ylim, +ax_p_ylim) 
              ax[i,1].set_ylim(-ax_v_ylim, +ax_v_ylim)

      if(MAKE_LEGEND):
          handles, labels = ax[0,0].get_legend_handles_labels()
          fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.suptitle('End-effector frame orientation and angular velocity', size=18)
      if(SHOW):
          plt.show()
      return fig, ax

  def plot_ocp_force(self, ocp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., 
                                                  MAKE_LEGEND=False, SHOW=True, AUTOSCALE=False):
      '''
      Plot OCP results (force)
      '''
      # Parameters
      N = ocp_data['T'] 
      dt = ocp_data['dt']
      # Extract EE traj
      f = np.array(ocp_data['fs'])
      f_ee_lin = f[:,:3]
      f_ee_ang = f[:,3:]
      # Get desired contact wrench (linear, angular)
      if('force_ref' in ocp_data.keys()):
          f_ee_ref = np.array(ocp_data['force_ref'])
      else:
          f_ee_ref = np.zeros((N,6))
      f_ee_lin_ref = f_ee_ref[:,:3]
      f_ee_ang_ref = f_ee_ref[:,3:]
      # Plots
      tspan = np.array([sum(ocp_data['dts'][:i]) for i in range(len(ocp_data['dts'])-1)]) #)] np.linspace(0, N*sum(ocp_data['dts']), N)
      if(ax is None or fig is None):
          fig, ax = plt.subplots(3, 2, sharex='col')
      if(label is None):
          label='End-effector force'
      xyz = ['x', 'y', 'z']
      for i in range(3):
          # Plot contact linear wrench (force) in LOCAL frame
          ax[i,0].plot(tspan, f_ee_lin[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot desired contact linear wrench (force) in LOCAL frame 
          if('force_ref' in ocp_data.keys()):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,0].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,0].plot(tspan, f_ee_lin_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)
          
          # Labels, tick labels+ grid
          ax[i,0].set_ylabel('$\\lambda^{lin}_%s$ (N)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,0].grid(True)

          # Plot contact angular wrench (torque) in LOCAL frame 
          ax[i,1].plot(tspan, f_ee_ang[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot desired contact anguler wrench (torque) in LOCAL frame
          if('force_ref' in ocp_data.keys()):
              handles, labels = ax[i,1].get_legend_handles_labels()
              if('reference' in labels):
                  handles.pop(labels.index('reference'))
                  ax[i,1].lines.pop(labels.index('reference'))
                  labels.remove('reference')
              ax[i,1].plot(tspan, f_ee_ang_ref[:,i], linestyle='-.', color='k', marker=None, label='reference', alpha=0.5)

          # Labels, tick labels+ grid
          ax[i,1].set_ylabel('$\\lambda^{ang}_%s$ (Nm)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,1].grid(True)
      
      # x-label + align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)

      # Set ylim if any
      if(AUTOSCALE):
          TOL = 1e-1
          ax_lin_ylim = 1.1*max(np.max(np.abs(f_ee_lin)), TOL)
          ax_ang_ylim = 1.1*max(np.max(np.abs(f_ee_ang)), TOL)
          for i in range(3):
              ax[i,0].set_ylim(f_ee_lin_ref[0,i]-ax_lin_ylim, f_ee_lin_ref[0,i]+ax_lin_ylim) 
              ax[i,1].set_ylim(f_ee_ang_ref[0,i]-ax_ang_ylim, f_ee_ang_ref[0,i]+ax_ang_ylim)

      if(MAKE_LEGEND):
          handles, labels = ax[0,0].get_legend_handles_labels()
          fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.suptitle('End-effector forces: linear and angular', size=18)
      if(SHOW):
          plt.show()
      return fig, ax




# Abstract MPC data handler : initialize, extract data + generate fancy plots
class MPCDataHandlerAbstract:
  '''
  Helper class to manage and plot data from MPC simulations
  '''

  def __init__(self, config, robot):

    self.__dict__ = config

    self.rmodel = robot.model
    self.rdata = robot.data

    self.nq = robot.model.nq
    self.nv = robot.model.nv
    self.nu = self.nq
    self.nx = self.nq + self.nv
    self.dts = [config['dt'] for i in range(config['N_h'])]
    self.dts.append(0.)
    
    self.check_config()

    # Check 1st contact name & reference frame in config file
    if(hasattr(self, 'contacts')):
        self.is_contact = True
        if(self.contacts[0]['pinocchioReferenceFrame'] == 'LOCAL'):
            self.PIN_REF_FRAME = pin.LOCAL
        else:
            self.PIN_REF_FRAME = pin.LOCAL_WORLD_ALIGNED
        self.contactFrameName = self.contacts[0]['contactModelFrameName']
        logger.warning("Contact force will be expressed in the "+str(self.PIN_REF_FRAME)+" convention")
    else:
        self.is_contact = False

  def check_attribute(self, attribute): 
    '''
    Check whether attribute exists and is well defined
    '''
    assert(type(attribute)==str), "Attribute to be checked must be a string"
    if(not hasattr(self, attribute)):
      logger.error("The MPC config parameter : "+str(attribute)+ " has not been defined ! Please correct the yaml config file.")

  def check_config(self):
    '''
    Check that config file is complete
    '''
    # general params
    self.check_attribute('simu_freq') #, int)
    self.check_attribute('ctrl_freq') #, int)
    self.check_attribute('plan_freq') #, int)
    self.check_attribute('T_tot')
    self.check_attribute('SAVE_DATA')
    self.check_attribute('RECORD_SOLVER_DATA')
    self.check_attribute('INIT_LOG')
    self.check_attribute('init_log_display_time')
    self.check_attribute('LOG')
    self.check_attribute('log_rate')
    self.check_attribute('WHICH_PLOTS')
    self.check_attribute('RICCATI')

    # actuation model stuff
    self.check_attribute('DELAY_SIM')
    self.check_attribute('DELAY_OCP')
    self.check_attribute('SCALE_TORQUES')
    self.check_attribute('NOISE_TORQUES')
    self.check_attribute('TORQUE_TRACKING')
    self.check_attribute('NOISE_STATE')
  
    # OCP stuff
    self.check_attribute('dt')
    self.check_attribute('WHICH_COSTS')


  # Save data (dict) into compressed npz
  def save_data(self, sim_data, save_name=None, save_dir=None):
      '''
      Saves data to a compressed npz file (binary)
      '''
      logger.info('Compressing & saving data...')
      if(save_name is None):
          save_name = 'sim_data_NO_NAME'+str(time.time())
      if(save_dir is None):
          save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
      save_path = save_dir+'/'+save_name+'.npz'
      np.savez_compressed(save_path, data=sim_data)
      logger.info("Saved data to "+str(save_path)+" !")
    

  # Allocate data and print config (base)
  def init_actuation_model(self):
    '''
    Initialize actuation model if necessary
    '''
    if(self.DELAY_OCP):
      self.check_attribute('delay_ocp_ms')
    if(self.DELAY_SIM):
      self.check_attribute('delay_sim_cycle')
    if(self.SCALE_TORQUES):
      self.check_attribute('alpha_min')
      self.check_attribute('alpha_max')
      self.check_attribute('beta_min')
      self.check_attribute('beta_max')
      self.alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max, size=(self.nq,))
      self.beta  = np.random.uniform(low=self.beta_min, high=self.beta_max, size=(self.nq,))
    if(self.NOISE_STATE):
      self.check_attribute('var_q')
      self.check_attribute('var_v')
      self.var_q = np.asarray(self.var_q)
      self.var_v = np.asarray(self.var_v)
    if(self.NOISE_TORQUES):
      self.check_attribute('var_u')
      self.var_u = 0.5*np.asarray(self.var_u) 
    if(self.TORQUE_TRACKING):
      self.check_attribute('Kp_low')
      self.check_attribute('Ki_low')
      self.check_attribute('Kd_low')
      self.gain_P = self.Kp_low*np.eye(self.nq)
      self.gain_I = self.Ki_low*np.eye(self.nq)
      self.gain_D = self.Kd_low*np.eye(self.nq)

  def init_solver_data(self):
    '''
    Allocate data for OCP solver stuff (useful to debug)
    '''
    self.K      = np.zeros((self.N_plan, self.N_h, self.nq, self.nx))     # Ricatti gains (K_0)
    self.Vxx    = np.zeros((self.N_plan, self.N_h+1, self.nx, self.nx)) # Hessian of the Value Function  
    self.Quu    = np.zeros((self.N_plan, self.N_h, self.nu, self.nu))   # Hessian of the Value Function 
    self.xreg   = np.zeros(self.N_plan)                                                   # State reg in solver (diag of Vxx)
    self.ureg   = np.zeros(self.N_plan)                                                   # Control reg in solver (diag of Quu)
    self.J_rank = np.zeros(self.N_plan)                                                 # Rank of Jacobian

  def init_cost_references(self):
    '''
    Allocate data for cost references to record
    '''
    if('ctrlReg' in self.WHICH_COSTS or 'ctrlRegGrav' in self.WHICH_COSTS):
      self.ctrl_ref       = np.zeros((self.N_plan, self.nu))
    if('stateReg' in self.WHICH_COSTS):
      self.state_ref      = np.zeros((self.N_plan, self.nx))
    if('translation' in self.WHICH_COSTS or 'placement' in self.WHICH_COSTS):
      self.lin_pos_ee_ref = np.zeros((self.N_plan, 3))
    if('velocity' in self.WHICH_COSTS):
      self.lin_vel_ee_ref = np.zeros((self.N_plan, 3))
      self.ang_vel_ee_ref = np.zeros((self.N_plan, 3))
    if('rotation' in self.WHICH_COSTS):
      self.ang_pos_ee_ref = np.zeros((self.N_plan, 3))
    if('force' in self.WHICH_COSTS):
      self.f_ee_ref       = np.zeros((self.N_plan, 6))

  def print_sim_params(self, sleep):
    '''
    Print out simulation parameters
    '''
    print('')
    print('                       *************************')
    print('                       ** Simulation is ready **') 
    print('                       *************************')        
    print("-------------------------------------------------------------------")
    print('- Total simulation duration            : T_tot           = '+str(self.T_tot)+' s')
    print('- Simulation frequency                 : f_simu          = '+str(float(self.simu_freq/1000.))+' kHz')
    print('- Control frequency                    : f_ctrl          = '+str(float(self.ctrl_freq/1000.))+' kHz')
    print('- Replanning frequency                 : f_plan          = '+str(float(self.plan_freq/1000.))+' kHz')
    print('- Total # of simulation steps          : N_simu          = '+str(self.N_simu))
    print('- Total # of control steps             : N_ctrl          = '+str(self.N_ctrl))
    print('- Total # of planning steps            : N_plan          = '+str(self.N_plan))
    print('- Duration of MPC horizon              : T_ocp           = '+str(self.T_h)+' s')
    print('- OCP integration step                 : dt              = '+str(self.dt)+' s')
    if(self.DELAY_SIM):
      print('- Simulate delay in low-level torque?  : DELAY_SIM       = '+str(self.DELAY_SIM)+' ('+str(self.delay_sim_cycle)+' cycles)')
    if(self.DELAY_OCP):
      print('- Simulate delay in OCP solution?      : DELAY_OCP       = '+str(self.DELAY_OCP)+' ('+str(self.delay_OCP_ms)+' ms)')
    print('- Affine scaling of ref. ctrl torque?  : SCALE_TORQUES   = '+str(self.SCALE_TORQUES))
    if(self.SCALE_TORQUES):
      print('    a = '+str(self.alpha)+'')
      print('    b = '+str(self.beta)+'')
    print('- Noise on torques?                    : NOISE_TORQUES   = '+str(self.NOISE_TORQUES))
    print('- Noise on state?                      : NOISE_STATE     = '+str(self.NOISE_STATE))
    print('- Low-level torque PI control ?        : TORQUE_TRACKING = '+str(self.TORQUE_TRACKING))
    print("-------------------------------------------------------------------")
    print('')
    time.sleep(sleep)

  # Allocate data (pure virtual)
  def init_predictions(self):
    raise NotImplementedError()
  
  def init_measurements(self):
    raise NotImplementedError()
  
  def init_sim_data(self):
    raise NotImplementedError()


  # Data recording helpers 
  def record_predictions(self):
    raise NotImplementedError()

  def record_plan_cycle_desired(self):
    raise NotImplementedError()

  def record_ctrl_cycle_desired(self):
    raise NotImplementedError()

  def record_simu_cycle_desired(self):
    raise NotImplementedError()


  def record_solver_data(self, nb_plan, ocpSolver):
    '''
    Handy function to record solver related data during MPC simulation
    '''
    if(self.RECORD_SOLVER_DATA):
      self.K[nb_plan, :, :, :]   = np.array(ocpSolver.K)         # Ricatti gains
      self.Vxx[nb_plan, :, :, :] = np.array(ocpSolver.Vxx)       # Hessians of V.F. 
      self.Quu[nb_plan, :, :, :] = np.array(ocpSolver.Quu)       # Hessians of Q 
      self.xreg[nb_plan]         = ocpSolver.x_reg               # Reg solver on x
      self.ureg[nb_plan]         = ocpSolver.u_reg               # Reg solver on u
      self.J_rank[nb_plan]       = np.linalg.matrix_rank(ocpSolver.problem.runningDatas[0].differential.pinocchio.J)

  def record_cost_references(self, nb_plan, ocpSolver):
    '''
    Handy function for MPC + clean plots
    Extract and record cost references of DAM into sim_data at i^th simulation step
     # careful, ref is hard-coded only for the first node
    '''
    # Get nodes
    m = ocpSolver.problem.runningModels[0]
    # Extract references and record
    if('ctrlReg' in self.WHICH_COSTS):
      self.ctrl_ref[nb_plan, :] = m.differential.costs.costs['ctrlReg'].cost.residual.reference
    if('ctrlRegGrav' in self.WHICH_COSTS):
      q = self.state_pred[nb_plan, 0, :self.nq]
      self.ctrl_ref[nb_plan, :] = pin_utils.get_u_grav(q, m.differential.pinocchio) #, self.armature)
    if('force' in self.WHICH_COSTS):
      if('force' in m.differential.costs.costs.todict().keys()):  
        self.f_ee_ref[nb_plan, :] = m.differential.costs.costs['force'].cost.residual.reference.vector
    if('stateReg' in self.WHICH_COSTS):
      self.state_ref[nb_plan, :] = m.differential.costs.costs['stateReg'].cost.residual.reference
    if('translation' in self.WHICH_COSTS):
      self.lin_pos_ee_ref[nb_plan, :] = m.differential.costs.costs['translation'].cost.residual.reference
    if('rotation' in self.WHICH_COSTS):
      self.ang_pos_ee_ref[nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['rotation'].cost.residual.reference)
    if('velocity' in self.WHICH_COSTS):
      self.lin_vel_ee_ref[nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[:3]
      self.ang_vel_ee_ref[nb_plan, :] = m.differential.costs.costs['velocity'].cost.residual.reference.vector[3:]
    if('placement' in self.WHICH_COSTS):
      self.lin_pos_ee_ref[nb_plan, :] = m.differential.costs.costs['placement'].cost.residual.reference.translation
      self.ang_pos_ee_ref[nb_plan, :] = pin.utils.matrixToRpy(m.differential.costs.costs['placement'].cost.residual.reference.rotation)

  # Extract data (pure virtual)
  def extract_data(self):
    raise NotImplementedError()
  
  def extract_solver_data(self):
    raise NotImplementedError()


  # Extract directly plot data from npz file 
  def extract_plot_data_from_npz(self, file, frame_of_interest):
    d = load_data(file)
    plot_data = self.extract_data(d, frame_of_interest)
    return plot_data


  # Virtual plotting functions (pure virtual)
  def plot_mpc_results(self):
    raise NotImplementedError()
  
  def plot_mpc_state(self):
    raise NotImplementedError()

  def plot_mpc_control(self):
    raise NotImplementedError()

  def plot_mpc_ricatti_diag(self):
    raise NotImplementedError()

  def plot_mpc_Vxx_eig(self):
    raise NotImplementedError()

  def plot_mpc_Vxx_diag(self):
    raise NotImplementedError()


  # Base plotting functions (base)
  def plot_mpc_endeff_linear(self, plot_data, PLOT_PREDICTIONS=False, 
                                pred_plot_sampling=100, 
                                SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                SHOW=True,
                                AUTOSCALE=False):
      '''
      Plot endeff data (linear position and velocity)
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''
      logger.info('Plotting end-eff data (linear)...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_ctrl = plot_data['N_ctrl']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot, N_simu+1)
      t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
      t_span_plan = np.linspace(0, T_tot, N_plan+1)
      fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
      # Plot endeff
      xyz = ['x', 'y', 'z']
      for i in range(3):

          if(PLOT_PREDICTIONS):
              lin_pos_ee_pred_i = plot_data['lin_pos_ee_pred'][:, :, i]
              lin_vel_ee_pred_i = plot_data['lin_vel_ee_pred'][:, :, i]
              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_x_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts']))]) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                  # Set up lists of (x,y) points for predicted positions
                  points_p = np.array([tspan_x_pred, lin_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  points_v = np.array([tspan_x_pred, lin_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                  segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                  lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                  lc_p.set_array(tspan_x_pred)
                  lc_v.set_array(tspan_x_pred)
                  # Customize
                  lc_p.set_linestyle('-')
                  lc_v.set_linestyle('-')
                  lc_p.set_linewidth(1)
                  lc_v.set_linewidth(1)
                  # Plot collections
                  ax[i,0].add_collection(lc_p)
                  ax[i,1].add_collection(lc_v)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                  my_colors = cm(colors)
                  ax[i,0].scatter(tspan_x_pred, lin_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                  ax[i,1].scatter(tspan_x_pred, lin_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

          # EE position
          ax[i,0].plot(t_span_plan, plot_data['lin_pos_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        #   ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.1)
          ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          # Plot reference
          if('translation' in plot_data['WHICH_COSTS']):
              ax[i,0].plot(t_span_plan[:-1], plot_data['lin_pos_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,0].set_ylabel('$P^{EE}_%s$  (m)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,0].grid(True)
          
          # EE velocity
          ax[i,1].plot(t_span_plan, plot_data['lin_vel_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        #   ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.1)
          ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          # Plot reference 
          if('velocity' in plot_data['WHICH_COSTS']):
              ax[i,1].plot(t_span_plan, [0.]*(N_plan+1), color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,1].set_ylabel('$V^{EE}_%s$  (m)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,1].grid(True)


      # Align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)
      # Set ylim if any
      TOL = 1e-3
      if(AUTOSCALE):
          ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['lin_pos_ee_mea'])), TOL)
          ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['lin_vel_ee_mea'])), TOL)
          for i in range(3):
              ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
              ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

      handles_p, labels_p = ax[0,0].get_legend_handles_labels()
      fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
      # Titles
      fig.suptitle('End-effector trajectories', size=18)
      # Save figs
      if(SAVE):
          figs = {'ee_lin': fig}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig, ax

  def plot_mpc_endeff_angular(self, plot_data, PLOT_PREDICTIONS=False, 
                                pred_plot_sampling=100, 
                                SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                SHOW=True,
                                AUTOSCALE=False):
      '''
      Plot endeff data (angular position and velocity)
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''
      logger.info('Plotting end-eff data (angular)...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_ctrl = plot_data['N_ctrl']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot, N_simu+1)
      t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
      t_span_plan = np.linspace(0, T_tot, N_plan+1)
      fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
      # Plot endeff
      xyz = ['x', 'y', 'z']
      for i in range(3):

          if(PLOT_PREDICTIONS):
              ang_pos_ee_pred_i = plot_data['ang_pos_ee_pred'][:, :, i]
              ang_vel_ee_pred_i = plot_data['ang_vel_ee_pred'][:, :, i]
              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_x_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts']))]) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                  # Set up lists of (x,y) points for predicted positions
                  points_p = np.array([tspan_x_pred, ang_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  points_v = np.array([tspan_x_pred, ang_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                  segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                  lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                  lc_p.set_array(tspan_x_pred)
                  lc_v.set_array(tspan_x_pred)
                  # Customize
                  lc_p.set_linestyle('-')
                  lc_v.set_linestyle('-')
                  lc_p.set_linewidth(1)
                  lc_v.set_linewidth(1)
                  # Plot collections
                  ax[i,0].add_collection(lc_p)
                  ax[i,1].add_collection(lc_v)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                  my_colors = cm(colors)
                  ax[i,0].scatter(tspan_x_pred, ang_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                  ax[i,1].scatter(tspan_x_pred, ang_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

          # EE position
          ax[i,0].plot(t_span_plan, plot_data['ang_pos_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
          ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.1)
          ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          # Plot reference
          if('rotation' in plot_data['WHICH_COSTS']):
              ax[i,0].plot(t_span_plan[:-1], plot_data['ang_pos_ee_ref'][:,i], 'm-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,0].set_ylabel('$RPY^{EE}_%s$  (m)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,0].grid(True)
          
          # EE velocity
          ax[i,1].plot(t_span_plan, plot_data['ang_vel_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
          ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.1)
          ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          # Plot reference 
          if('velocity' in plot_data['WHICH_COSTS']):
              ax[i,1].plot(t_span_plan, [0.]*(N_plan+1), 'm-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,1].set_ylabel('$W^{EE}_%s$  (m)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,1].grid(True)


      # Align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)
      # Set ylim if any
      TOL = 1e-3
      if(AUTOSCALE):
          ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['ang_pos_ee_mea'])), TOL)
          ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['ang_vel_ee_mea'])), TOL)
          for i in range(3):
              ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
              ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

      handles_p, labels_p = ax[0,0].get_legend_handles_labels()
      fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
      # Titles
      fig.suptitle('End-effector frame orientation (RPY) and angular velocity', size=18)
      # Save figs
      if(SAVE):
          figs = {'ee_ang': fig}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig, ax

  def plot_mpc_force(self, plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
      '''
      Plot EE force data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''
      logger.info('Plotting force data...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_ctrl = plot_data['N_ctrl']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      dt_simu = plot_data['dt_simu']
      dt_ctrl = plot_data['dt_ctrl']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot - dt_simu, N_simu)
      t_span_ctrl = np.linspace(0, T_tot - dt_ctrl, N_ctrl)
      t_span_plan = np.linspace(0, T_tot - dt_plan, N_plan)
      fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
      # Plot endeff
      xyz = ['x', 'y', 'z']
      for i in range(3):

          if(PLOT_PREDICTIONS):
              f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_x_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts'])-1)]) #np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                  # Set up lists of (x,y) points for predicted positions
                  points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
                  lc_f.set_array(tspan_x_pred)
                  # Customize
                  lc_f.set_linestyle('-')
                  lc_f.set_linewidth(1)
                  # Plot collections
                  ax[i,0].add_collection(lc_f)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h-1), 1] 
                  my_colors = cm(colors)
                  ax[i,0].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
        
          # EE linear force
          ax[i,0].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
          ax[i,0].plot(t_span_simu, plot_data['f_ee_mea'][:,i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        #   ax[i,0].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,i], 'r-', label='Measured', linewidth=2)
          # Plot reference
          if('force' in plot_data['WHICH_COSTS']):
              ax[i,0].plot(t_span_plan, plot_data['f_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,0].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,0].grid(True)

          # EE angular force 
          ax[i,1].plot(t_span_plan, plot_data['f_ee_des_PLAN'][:,3+i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
          ax[i,1].plot(t_span_simu, plot_data['f_ee_mea'][:,3+i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        #   ax[i,1].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,3+i]-[plot_data['f_ee_ref'][3+i]]*(N_simu+1), 'r-', label='Measured', linewidth=2)
          # Plot reference
          if('force' in plot_data['WHICH_COSTS']):
              ax[i,1].plot(t_span_plan, plot_data['f_ee_ref'][:,3+i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i,1].set_ylabel('$\\tau^{EE}_%s$  (Nm)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,1].grid(True)
      
      # Align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)
      # Set ylim if any
      TOL = 1e-3
      if(AUTOSCALE):
        #   ax_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_pred'])), TOL) # 1.1*max( np.nanmax(np.abs(plot_data['f_ee_mea'])), TOL )
        #   ax_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_pred'])), TOL) # 1.1*max( np.nanmax(np.abs(plot_data['f_ee_mea'])), TOL )
          for i in range(3):
            #   ax[i,0].set_ylim(-ax_ylim, ax_ylim) 
            #   ax[i,1].set_ylim(-ax_ylim, ax_ylim) 
              ax[i,0].set_ylim(-50, 50) 
              ax[i,1].set_ylim(-50, 50) 

      handles_p, labels_p = ax[0,0].get_legend_handles_labels()
      fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
      # Titles
      fig.suptitle('End-effector forces (LOCAL)', size=18)
      # Save figs
      if(SAVE):
          figs = {'f': fig}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig, ax

  def plot_mpc_ricatti_svd(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                              SHOW=True):
      '''
      Plot ricatti data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting Ricatti singular values...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_K, ax_K = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
      # For each joint
      for i in range(nq):
          # Ricatti gains singular values
          ax_K[i].plot(t_span_plan_u, plot_data['K_svd'][:, 0, i], 'b-', label='Singular Values of Ricatti gain K')
          ax_K[i].set_ylabel('$\sigma_{}$'.format(i), fontsize=12)
          ax_K[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_K[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_K[i].grid(True)
          # Set xlabel on bottom plot
          if(i == nq-1):
              ax_K[i].set_xlabel('t (s)', fontsize=16)
      # y axis labels
      # fig_K.text(0.04, 0.5, 'Singular values', va='center', rotation='vertical', fontsize=16)
      # Titles
      fig_K.suptitle('Singular Values of Ricatti feedback gains K', size=16)
      # Save figs
      if(SAVE):
          figs = {'K_svd': fig_K}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_K

  def plot_mpc_Quu_eig(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
      '''
      Plot Quu eigenvalues
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting Quu eigenvalues...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_Q, ax_Q = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
      # For each state
      for i in range(nq):
          # Quu eigenvals
          ax_Q[i].plot(t_span_plan_u, plot_data['Quu_eig'][:, 0, i], 'b-', label='Quu eigenvalue')
          ax_Q[i].set_ylabel('$\lambda_{}$'.format(i), fontsize=12)
          ax_Q[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_Q[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_Q[i].grid(True)
          # Set xlabel on bottom plot
          if(i == nq-1):
              ax_Q[i].set_xlabel('t (s)', fontsize=16)
      # Titles
      fig_Q.suptitle('Eigenvalues of Hamiltonian Hessian Quu', size=16)
      # Save figs
      if(SAVE):
          figs = {'Q_eig': fig_Q}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_Q

  def plot_mpc_Quu_diag(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
      '''
      Plot Quu diagonal terms
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting Quu diagonal...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_Q, ax_Q = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
      # For each state
      for i in range(nq):
          # Quu diag
          ax_Q[i].plot(t_span_plan_u, plot_data['Quu_diag'][:, 0, i], 'b-', label='Quu diagonal')
          ax_Q[i].set_ylabel('$Quu_{}$'.format(i), fontsize=12)
          ax_Q[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_Q[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_Q[i].grid(True)
          # Set xlabel on bottom plot
          if(i == nq-1):
              ax_Q[i].set_xlabel('t (s)', fontsize=16)
      # Titles
      fig_Q.suptitle('Diagonal of Hamiltonian Hessian Quu', size=16)
      # Save figs
      if(SAVE):
          figs = {'Q_diag': fig_Q}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_Q

  def plot_mpc_solver(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True):
      '''
      Plot solver data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting solver data...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_S, ax_S = plt.subplots(2, 1, figsize=(19.2,10.8), sharex='col') 
      # Xreg
      ax_S[0].plot(t_span_plan_u, plot_data['xreg'], 'b-', label='xreg')
      ax_S[0].set(xlabel='t (s)', ylabel='$xreg$')
      ax_S[0].grid(True)
      # Ureg
      ax_S[1].plot(t_span_plan_u, plot_data['ureg'], 'r-', label='ureg')
      ax_S[1].set(xlabel='t (s)', ylabel='$ureg$')
      ax_S[1].grid(True)

      # Titles
      fig_S.suptitle('OCP solver regularization on x (Vxx diag) and u (Quu diag)', size=16)
      # Save figs
      if(SAVE):
          figs = {'S': fig_S}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_S

  def plot_mpc_jacobian(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                              SHOW=True):
      '''
      Plot jacobian data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting solver data...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_J, ax_J = plt.subplots(1, 1, figsize=(19.2,10.8), sharex='col') 
      # Rank of Jacobian
      ax_J.plot(t_span_plan_u, plot_data['J_rank'], 'b-', label='rank')
      ax_J.set(xlabel='t (s)', ylabel='rank')
      ax_J.grid(True)

      # Titles
      fig_J.suptitle('Rank of Jacobian J(q)', size=16)
      # Save figs
      if(SAVE):
          figs = {'J': fig_J}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_J



