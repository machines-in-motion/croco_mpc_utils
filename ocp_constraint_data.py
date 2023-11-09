
"""
@package croco_mpc_utils
@file ocp_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Data handlers for classical OCP wrapper class 
"""

import numpy as np
import pinocchio as pin

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical


# Classical OCP data handler : extract data + generate fancy plots
class OCPDataHandlerClassicalWithConstraints(OCPDataHandlerClassical):

  def __init__(self, ocp):
    super().__init__(ocp)

  def extract_data(self, xs, us): 
    ocp_data = super().extract_data(xs, us) 
    # Add constraints
    self.find_constraints()
    return ocp_data
  
  def find_constraints(self):
      '''
      Detects the type of constraints defined in the OCP
      '''
      cstrNames = []
      for m in self.ocp.runningModels:
        if(hasattr(m.differential, 'constraints')):
          for cstr_name in m.differential.constraints.todict().keys():
            if(cstr_name not in cstrNames):
              cstrNames.append(cstr_name)
      logger.warning("Found "+str(len(cstrNames))+" constraints names in the cost function !")
      if(len(cstrNames) > 1):
        logger.warning("Found "+str(len(cstrNames))+" constraints names in the cost function !")
        logger.warning(str(cstrNames))
      if(len(cstrNames) == 0):
        logger.warning("Did not find any constraints name id.")
      return cstrNames

  def plot_ocp_results(self, OCP_DATA, which_plots='all', labels=None, markers=None, colors=None, sampling_plot=1, SHOW=False):
      '''
      Plot OCP results from 1 or several OCP solvers
          X, U, EE trajs
          INPUT 
          OCP_DATA         : OCP data or list of OCP data (cf. data_utils.extract_ocp_data())
      '''
      logger.info("Plotting OCP solver data...")
      if(type(OCP_DATA) != list):
          OCP_DATA = [OCP_DATA]
      if(labels==None):
          labels=[None for k in range(len(OCP_DATA))]
      if(markers==None):
          markers=[None for k in range(len(OCP_DATA))]
      if(colors==None):
          colors=[None for k in range(len(OCP_DATA))]
      for k,data in enumerate(OCP_DATA):
          # If last plot, make legend
          make_legend = False
          if(k+sampling_plot > len(OCP_DATA)-1):
              make_legend=True
          # Return figs and axes object in case need to overlay new plots
          if(k==0):
              if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('xs' in data.keys()):
                      fig_x, ax_x = self.plot_ocp_state(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('us' in data.keys()):
                      fig_u, ax_u = self.plot_ocp_control(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('xs' in data.keys()):
                      fig_ee_lin, ax_ee_lin = self.plot_ocp_endeff_linear(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                      fig_ee_ang, ax_ee_ang = self.plot_ocp_endeff_angular(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
              if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                  if('fs' in data.keys()):
                      fig_f, ax_f = self.plot_ocp_force(data, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False, AUTOSCALE=True)
          else:
              if(k%sampling_plot==0):
                  if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('xs' in data.keys()):
                          self.plot_ocp_state(data, fig=fig_x, ax=ax_x, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('us' in data.keys()):
                          self.plot_ocp_control(data, fig=fig_u, ax=ax_u, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('xs' in data.keys()):
                          self.plot_ocp_endeff_linear(data, fig=fig_ee_lin, ax=ax_ee_lin, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False)
                  if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
                      if('fs' in data.keys()):
                          self.plot_ocp_force(data, fig=fig_f, ax=ax_f, label=labels[k], marker=markers[k], color=colors[k], MAKE_LEGEND=make_legend, SHOW=False, AUTOSCALE=True)
      if(SHOW):
          plt.show()
      
      # Record and return if user needs to overlay stuff
      fig = {}
      ax = {}
      if('x' in which_plots or which_plots =='all' or 'all' in which_plots):
          if('xs' in data.keys()):
              fig['x'] = fig_x
              ax['x'] = ax_x
      if('u' in which_plots or which_plots =='all' or 'all' in which_plots):
          if('us' in data.keys()):
              fig['u'] = fig_u
              ax['u'] = ax_u
      if('ee' in which_plots or which_plots =='all' or 'all' in which_plots):
          if('xs' in data.keys()):
              fig['ee_lin'] = fig_ee_lin
              ax['ee_lin'] = ax_ee_lin
              fig['ee_ang'] = fig_ee_ang
              ax['ee_ang'] = ax_ee_ang
      if('f' in which_plots or which_plots =='all' or 'all' in which_plots):
          if('fs' in data.keys()):
              fig['f'] = fig_f
              ax['f'] = ax_f

      return fig, ax
  
  def plot_ocp_state(self, ocp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
      '''
      Plot OCP results (state)
      '''
      # Parameters
      N = ocp_data['T'] 
      dt = ocp_data['dt']
      nq = ocp_data['nq'] 
      nv = ocp_data['nv'] 
      # Extract trajectories
      x = np.array(ocp_data['xs'])
      q = x[:,:nq]
      v = x[:,nq:nq+nv]
      # If state reg cost, 
      if('stateReg' in ocp_data['active_costs']):
          x_reg_ref = np.array(ocp_data['stateReg_ref'])
      # Plots
      tspan = np.array([sum(ocp_data['dts'][:i]) for i in range(len(ocp_data['dts']))]) #np.linspace(0, N*dt, N+1)
      if(ax is None or fig is None):
          fig, ax = plt.subplots(nq, 2, sharex='col') 
      if(label is None):
          label='State'
      for i in range(nq):
          # Plot positions
          ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot joint position regularization reference
          if('stateReg' in ocp_data['active_costs']):
              handles, labels = ax[i,0].get_legend_handles_labels()
              if('reg_ref' in labels):
                  handles.pop(labels.index('reg_ref'))
                  ax[i,0].lines.pop(labels.index('reg_ref'))
                  labels.remove('reg_ref')
              ax[i,0].plot(tspan, x_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
          ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,0].grid(True)
      for i in range(nv):
          # Plot velocities
          ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)  

          # Plot joint velocity regularization reference
          if('stateReg' in ocp_data['active_costs']):
              handles, labels = ax[i,1].get_legend_handles_labels()
              if('reg_ref' in labels):
                  handles.pop(labels.index('reg_ref'))
                  ax[i,1].lines.pop(labels.index('reg_ref'))
                  labels.remove('reg_ref')
              ax[i,1].plot(tspan, x_reg_ref[:,nq+i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
          
          # Labels, tick labels and grid
          ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i,1].grid(True)  

          # Set ylim if any
        #   ax[i,0].set_ylim(ocp_data['pin_model'].lowerPositionLimit[i], ocp_data['pin_model'].upperPositionLimit[i]) 
        #   ax[i,1].set_ylim(-ocp_data['pin_model'].velocityLimit[i], ocp_data['pin_model'].velocityLimit[i])


      # Common x-labels + align
      ax[-1,0].set_xlabel('Time (s)', fontsize=16)
      ax[-1,1].set_xlabel('Time (s)', fontsize=16)
      fig.align_ylabels(ax[:, 0])
      fig.align_ylabels(ax[:, 1])


      if(MAKE_LEGEND):
          handles, labels = ax[0,0].get_legend_handles_labels()
          fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.align_ylabels()
      fig.suptitle('State trajectories', size=18)
      if(SHOW):
          plt.show()
      return fig, ax

  def plot_ocp_control(self, ocp_data, fig=None, ax=None, label=None, marker=None, color=None, alpha=1., MAKE_LEGEND=False, SHOW=True):
      '''
      Plot OCP results (control)
      '''
      # Parameters
      N = ocp_data['T'] 
      dt = ocp_data['dt']
      nu = ocp_data['nu'] 
      # Extract trajectory
      u = np.array(ocp_data['us'])
      if('ctrlReg' in ocp_data['active_costs']):
          ureg_ref  = np.array(ocp_data['ctrlReg_ref']) 
      if('ctrlRegGrav' in ocp_data['active_costs']):
          ureg_grav = np.array(ocp_data['ctrlRegGrav_ref'])

      tspan = np.array([sum(ocp_data['dts'][:i]) for i in range(len(ocp_data['dts'])-1)]) #np.linspace(0, N*dt-dt, N)
      if(ax is None or fig is None):
          fig, ax = plt.subplots(nu, 1, sharex='col') 
      if(label is None):
          label='Control'    

      for i in range(nu):
          # Plot optimal control trajectory
          ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label, color=color, alpha=alpha)

          # Plot control regularization reference 
          if('ctrlReg' in ocp_data['active_costs']):
              handles, labels = ax[i].get_legend_handles_labels()
              if('u_reg' in labels):
                  handles.pop(labels.index('u_reg'))
                  ax[i].lines.pop(labels.index('u_reg'))
                  labels.remove('u_reg')
              ax[i].plot(tspan, ureg_ref[:,i], linestyle='-.', color='k', marker=None, label='u_reg', alpha=0.5)

          # Plot gravity compensation torque
          if('ctrlRegGrav' in ocp_data['active_costs']):
              handles, labels = ax[i].get_legend_handles_labels()
              if('grav(q)' in labels):
                  handles.pop(labels.index('u_grav(q)'))
                  ax[i].lines.pop(labels.index('u_grav(q)'))
                  labels.remove('u_grav(q)')
              ax[i].plot(tspan, ureg_grav[:,i], linestyle='-.', color='m', marker=None, label='u_grav(q)', alpha=0.5)
          
          # Labels, tick labels, grid
          ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
          ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax[i].grid(True)

      # Set x label + align
      ax[-1].set_xlabel('Time (s)', fontsize=16)
      fig.align_ylabels(ax[:])
      # Legend
      if(MAKE_LEGEND):
          handles, labels = ax[i].get_legend_handles_labels()
          fig.legend(handles, labels, loc='upper right', prop={'size': 16})
      fig.suptitle('Control trajectories', size=18)
      if(SHOW):
          plt.show()
      return fig, ax
