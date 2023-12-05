
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
from croco_mpc_utils.ocp_core_data import OCPDataHandlerAbstract, MPCDataHandlerAbstract


# Classical OCP data handler : extract data + generate fancy plots
class OCPDataHandlerClassical(OCPDataHandlerAbstract):

  def __init__(self, ocp):
    super().__init__(ocp)

  def extract_data(self, xs, us): #, ee_frame_name, ct_frame_name):
    return super().extract_data(xs, us) #, ee_frame_name, ct_frame_name)

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




# Classical MPC data handler : initialize, extract data + generate fancy plots
class MPCDataHandlerClassical(MPCDataHandlerAbstract):

  def __init__(self, config, robot):
    super().__init__(config, robot)

  # Allocate data 
  def init_predictions(self):
    '''
    Allocate data for state, control & force predictions
    '''
    self.state_pred     = np.zeros((self.N_plan, self.N_h+1, self.nx)) # Predicted states  ( self.ocp.xs : {x* = (q*, v*)} )
    self.ctrl_pred      = np.zeros((self.N_plan, self.N_h, self.nu))   # Predicted torques ( self.ocp.us : {u*} )
    self.force_pred     = np.zeros((self.N_plan, self.N_h, 6))         # Predicted EE contact forces
    self.state_des_PLAN = np.zeros((self.N_plan+1, self.nx))           # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    self.ctrl_des_PLAN  = np.zeros((self.N_plan, self.nu))             # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    self.force_des_PLAN = np.zeros((self.N_plan, 6))                   # Predicted EE contact forces planner frequency  
    self.state_des_CTRL = np.zeros((self.N_ctrl+1, self.nx))           # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    self.ctrl_des_CTRL  = np.zeros((self.N_ctrl, self.nu))             # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    self.force_des_CTRL = np.zeros((self.N_ctrl, 6))                   # Reference EE contact force at motor drivers freq
    self.state_des_SIMU = np.zeros((self.N_simu+1, self.nx))           # Reference state at actuation freq ( x* interpolated at SIMU freq )
    self.ctrl_des_SIMU  = np.zeros((self.N_simu, self.nu))             # Reference input at actuation freq ( u* interpolated at SIMU freq )
    self.force_des_SIMU = np.zeros((self.N_simu, 6))                   # Reference EE contact force at actuation freq

  def init_measurements(self, x0):
    '''
    Allocate data for simulation state & force measurements 
    '''
    self.state_mea_SIMU                = np.zeros((self.N_simu+1, self.nx))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    self.state_mea_no_noise_SIMU       = np.zeros((self.N_simu+1, self.nx))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    self.force_mea_SIMU                = np.zeros((self.N_simu, 6))           # Measured contact forces
    self.tau_mea_SIMU                  = np.zeros((self.N_simu, self.nu))     # Measured torque 
    self.tau_mea_derivative_SIMU       = np.zeros((self.N_simu, self.nu))     # Measured torque derivative
    self.state_mea_SIMU[0, :]          = x0
    self.state_mea_no_noise_SIMU[0, :] = x0
    
  def init_sim_data(self, x0):
    '''
    Allocate and initialize MPC simulation data
    '''
    # MPC & simulation parameters
    self.N_plan = int(self.T_tot*self.plan_freq)         # Total number of planning steps in the simulation
    self.N_ctrl = int(self.T_tot*self.ctrl_freq)         # Total number of control steps in the simulation 
    self.N_simu = int(self.T_tot*self.simu_freq)         # Total number of simulation steps 
    self.T_h = self.N_h*self.dt                          # Duration of the MPC horizon (s)
    self.dt_ctrl = float(1./self.ctrl_freq)              # Duration of 1 control cycle (s)
    self.dt_plan = float(1./self.plan_freq)              # Duration of 1 planning cycle (s)
    self.dt_simu = float(1./self.simu_freq)              # Duration of 1 simulation cycle (s)
    self.OCP_TO_PLAN_RATIO = self.dt_plan / self.dt      # Ratio of replanning interval over OCP interval
    # Init actuation model
    self.init_actuation_model()
    # Cost references 
    self.init_cost_references()
    # Predictions
    self.init_predictions()
    # Measurements
    self.init_measurements(x0)

    # OCP solver-specific data
    if(self.RECORD_SOLVER_DATA):
      self.init_solver_data()
   
    logger.info("Initialized MPC simulation data.")

    if(self.INIT_LOG):
      self.print_sim_params(self.init_log_display_time)



  def record_predictions(self, nb_plan, ocpSolver):
    '''
    Records the MPC predictions at the current step (state, control and forces if contact is specified)
    Input:
      nb_plan   : MPC (a.k.a. replanning) cycle numer
      ocpSolver : solver object (derived from crocoddy.SolverAbstract)
    '''
    # State and control predictions
    self.state_pred[nb_plan, :, :] = np.array(ocpSolver.xs)
    self.ctrl_pred[nb_plan, :, :]  = np.array(ocpSolver.us)
    # Extract current state & control + first state prediction
    self.x_curr = self.state_pred[nb_plan, 0, :]    # x0* = measured state    (q0*, v0*) = (q^mea,  v^mea)
    self.x_pred = self.state_pred[nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*)  
    self.u_curr = self.ctrl_pred[nb_plan, 0, :]     # u0* = optimal control   
    # Record forces predictions in the right frame + extract current & next force
    if(self.is_contact):
        id_endeff = self.rmodel.getFrameId(self.contactFrameName)
        jMf = self.rmodel.frames[id_endeff].placement
        lwaMf = pin.SE3(self.rdata.oMf[id_endeff].rotation, np.zeros(3))
        if(self.PIN_REF_FRAME == pin.LOCAL):
            self.force_pred[nb_plan, :, :] = \
                np.array([jMf.actInv(ocpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].fext).vector for i in range(self.N_h)])
        elif(self.PIN_REF_FRAME == pin.LOCAL_WORLD_ALIGNED or self.PIN_REF_FRAME == pin.WORLD):
            self.force_pred[nb_plan, :, :] = \
                np.array([lwaMf.act(jMf.actInv(ocpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].fext)).vector for i in range(self.N_h)])
        else:
            logger.error("The Pinocchio reference frame must be in ['LOCAL', LOCAL_WORLD_ALIGNED', 'WORLD']")
        self.f_curr = self.force_pred[nb_plan, 0, :]
        self.f_pred = self.force_pred[nb_plan, 1, :]

  def record_simu_cycle_measured(self, nb_simu, x_mea_SIMU, x_mea_no_noise_SIMU, tau_mea_SIMU, f_mea_SIMU):
    '''
    Records the measurements of state, torque and contact forces at the current simulation cycle
     Input:
      nb_simu             : simulation cycle number
      x_mea_SIMU          : measured position-velocity state from rigid-body physics simulator
      x_mea_no_noise_SIMU :  " " without sensing noise
      tau_mea_SIMU        : torque measured after actuation/transmission effects
      f_mea_SIMU          : external contact wrench measured from simulator 
    NOTE: this fucntion also computes the derivatives of the joint torques 
    '''
    self.state_mea_no_noise_SIMU[nb_simu+1, :] = x_mea_SIMU
    self.state_mea_SIMU[nb_simu+1, :]          = x_mea_no_noise_SIMU
    self.force_mea_SIMU[nb_simu, :]            = f_mea_SIMU
    self.tau_mea_SIMU[nb_simu, :]              = tau_mea_SIMU
    if(nb_simu > 0):
        self.tau_mea_derivative_SIMU[nb_simu, :] = (tau_mea_SIMU - self.tau_mea_SIMU[nb_simu-1, :])/self.dt_simu


  def record_plan_cycle_desired(self, nb_plan):
    '''
    Records the desired (state, control, force) at the planning (a.k.a. MPC) frequency
    - The desired state and force are interpolated bewteen the current (a.k.a. x0*, f0*) 
      and the predicted (a.k.a. x1*, f1*) ones from the OCP sampling frequency to the MPC (a.k.a. planning) frequency
    - The desired control is set to the current (a.k.a. last computed u0*) optimal control - NOT interpolated 
    Input:
      nb_plan   : mpc (a.k.a. replanning) cycle numer
    '''
    if(nb_plan==0):
        self.state_des_PLAN[nb_plan, :] = self.x_curr  
    self.ctrl_des_PLAN[nb_plan, :]      = self.u_curr   
    self.state_des_PLAN[nb_plan+1, :]   = self.x_curr + self.OCP_TO_PLAN_RATIO * (self.x_pred - self.x_curr)    
    if(self.is_contact):
        self.force_des_PLAN[nb_plan, :] = self.f_curr + self.OCP_TO_PLAN_RATIO * (self.f_pred - self.f_curr)    


  # Extract MPC simu-specific plotting data from sim data
  def extract_data(self, frame_of_interest):
    '''
    Extract plot data from simu data
    '''
    logger.info('Extracting plot data from simulation data...')
    
    plot_data = self.__dict__.copy()
    # Get costs
    plot_data['WHICH_COSTS'] = self.WHICH_COSTS
    plot_data['dts'] = self.dts
    # Robot model & params
    plot_data['pin_model'] = self.rmodel
    self.id_endeff = self.rmodel.getFrameId(frame_of_interest)
    nq = self.nq ; nv = self.nv ; 
    # Control predictions
    plot_data['u_pred'] = self.ctrl_pred
    plot_data['u_des_PLAN'] = self.ctrl_des_PLAN
    # State predictions (at PLAN freq)
    plot_data['q_pred']     = self.state_pred[:,:,:nq]
    plot_data['v_pred']     = self.state_pred[:,:,nq:nq+nv]
    plot_data['q_des_PLAN'] = self.state_des_PLAN[:,:nq]
    plot_data['v_des_PLAN'] = self.state_des_PLAN[:,nq:nq+nv] 
    # State measurements (at SIMU freq)
    plot_data['q_mea']          = self.state_mea_SIMU[:,:nq]
    plot_data['v_mea']          = self.state_mea_SIMU[:,nq:nq+nv]
    plot_data['q_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,:nq]
    plot_data['v_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,nq:nq+nv]
    # Torque measurements
    plot_data['u_mea'] = self.tau_mea_SIMU
    # Extract gravity torques
    plot_data['grav'] = np.zeros((self.N_simu+1, nq))
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'])
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    for node_id in range(self.N_h+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea']          = pin_utils.get_p_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_mea']          = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
      # Angular
    plot_data['ang_pos_ee_mea']          = pin_utils.get_rpy_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_mea']          = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
      # Angular
    plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
    # Extract EE force
    plot_data['f_ee_pred']     = self.force_pred
    plot_data['f_ee_mea']      = self.force_mea_SIMU
    plot_data['f_ee_des_PLAN'] = self.force_des_PLAN

    # Solver data (optional)
    if(self.RECORD_SOLVER_DATA):
      self.extract_solver_data(plot_data)
    
    return plot_data
    
  def extract_solver_data(self, plot_data):
    # nq = self.nq ; nv = self.nv ; nu = nq ; nx = nq + nv
    # # Get SVD & diagonal of Ricatti + record in sim data
    # plot_data['K_svd'] = np.zeros((self.N_plan, self.N_h, nq))
    # plot_data['Kp_diag'] = np.zeros((self.N_plan, self.N_h, nq))
    # plot_data['Kv_diag'] = np.zeros((self.N_plan, self.N_h, nv))
    # plot_data['Ktau_diag'] = np.zeros((self.N_plan, self.N_h, nu))
    # for i in range(self.N_plan):
    #   for j in range(self.N_h):
    #     plot_data['Kp_diag'][i, j, :] = self.K[i, j, :, :nq].diagonal()
    #     plot_data['Kv_diag'][i, j, :] = self.K[i, j, :, nq:nq+nv].diagonal()
    #     plot_data['Ktau_diag'][i, j, :] = self.K[i, j, :, -nu:].diagonal()
    #     _, sv, _ = np.linalg.svd(self.K[i, j, :, :])
    #     plot_data['K_svd'][i, j, :] = np.sort(sv)[::-1]
    # # Get diagonal and eigenvals of Vxx + record in sim data
    # plot_data['Vxx_diag'] = np.zeros((self.N_plan,self.N_h+1, nx))
    # plot_data['Vxx_eig'] = np.zeros((self.N_plan, self.N_h+1, nx))
    # for i in range(self.N_plan):
    #   for j in range(self.N_h+1):
    #     plot_data['Vxx_diag'][i, j, :] = self.Vxx[i, j, :, :].diagonal()
    #     plot_data['Vxx_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Vxx[i, j, :, :]))[::-1]
    # # Get diagonal and eigenvals of Quu + record in sim data
    # plot_data['Quu_diag'] = np.zeros((self.N_plan,self.N_h, nu))
    # plot_data['Quu_eig'] = np.zeros((self.N_plan, self.N_h, nu))
    # for i in range(self.N_plan):
    #   for j in range(self.N_h):
    #     plot_data['Quu_diag'][i, j, :] = self.Quu[i, j, :, :].diagonal()
    #     plot_data['Quu_eig'][i, j, :] = np.sort(np.linalg.eigvals(self.Quu[i, j, :, :]))[::-1]
    # # Get Jacobian
    # plot_data['J_rank'] = self.J_rank
    # # Get solve regs
    # plot_data['xreg'] = self.xreg
    # plot_data['ureg'] = self.ureg
    plot_data['iter'] = self.iter
    plot_data['KKT']  = self.KKT
    plot_data['maxiter']  = self.maxiter
    plot_data['solver_termination_tolerance']  = self.solver_termination_tolerance
  
  # Plot data - classical OCP specific plotting functions
  def plot_mpc_results(self, plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                                pred_plot_sampling=100, 
                                                SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                                SHOW=True,
                                                AUTOSCALE=False):
      '''
      Plot sim data
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

      plots = {}

      if('state' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['state'] = self.plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)
      
      if('control' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['control'] = self.plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)

      if('ee' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['ee_lin'] = self.plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)
          plots['ee_ang'] = self.plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)

      if('force' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['force'] = self.plot_mpc_force(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)

      if('K' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('K_diag' in plot_data.keys()):
              plots['K_diag'] = self.plot_mpc_ricatti_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)
          if('K_svd' in plot_data.keys()):
              plots['K_svd'] = self.plot_mpc_ricatti_svd(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('V' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('V_diag' in plot_data.keys()):
              plots['V_diag'] = self.plot_mpc_Vxx_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
          if('V_eig' in plot_data.keys()):
              plots['V_eig'] = self.plot_mpc_Vxx_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)

      if('S' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('S' in plot_data.keys()):
              plots['S'] = self.plot_mpc_solver_reg(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('J' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('J' in plot_data.keys()):
              plots['J'] = self.plot_mpc_jacobian(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)

      if('Q' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          if('Q_diag' in plot_data.keys()):
              plots['Q_diag'] = self.plot_mpc_Quu_diag(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
          if('Q_eig' in plot_data.keys()):
              plots['Q_eig'] = self.plot_mpc_Quu_eig(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False)
      if('solver' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['solver'] = self.plot_mpc_solver(plot_data, SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                                  SHOW=False)
      if(SHOW):
          plt.show() 

  def plot_mpc_state(self, plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True, AUTOSCALE=False):
      '''
      Plot state data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting state data...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']
      nx = plot_data['nx']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot, N_simu+1)
      t_span_plan = np.linspace(0, T_tot, N_plan+1)
      fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
      # For each joint
      for i in range(nq):

          if(PLOT_PREDICTIONS):

              # Extract state predictions of i^th joint
              q_pred_i = plot_data['q_pred'][:,:,i]
              v_pred_i = plot_data['v_pred'][:,:,i]
              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_x_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts']))]) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                  tspan_u_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts'])-1)]) #np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                  # Set up lists of (x,y) points for predicted positions and velocities
                  points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                  segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                  lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                  lc_q.set_array(tspan_x_pred)
                  lc_v.set_array(tspan_x_pred) 
                  # Customize
                  lc_q.set_linestyle('-')
                  lc_v.set_linestyle('-')
                  lc_q.set_linewidth(1)
                  lc_v.set_linewidth(1)
                  # Plot collections
                  ax_x[i,0].add_collection(lc_q)
                  ax_x[i,1].add_collection(lc_v)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                  my_colors = cm(colors)
                  ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                  ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',

          # Joint position
          ax_x[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        #   ax_x[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r', label='Measured', linewidth=1, alpha=0.1)
          ax_x[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          # Plot joint position regularization reference
          if('stateReg' in plot_data['WHICH_COSTS']):
              ax_x[i,0].plot(t_span_plan[:-1], plot_data['state_ref'][:, i], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
          ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
          ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax_x[i,0].grid(True)
          # Set xlim if any
          if(AUTOSCALE):
            ax_x[i,0].set_ylim(plot_data['pin_model'].lowerPositionLimit[i], plot_data['pin_model'].upperPositionLimit[i]) 

          # Joint velocity 
          ax_x[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN)', alpha=0.5)
        #   ax_x[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r', label='Measured', linewidth=1, alpha=0.1)
          ax_x[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured (no noise)', alpha=0.6)
          if('stateReg' in plot_data['WHICH_COSTS']):
              ax_x[i,1].plot(t_span_plan[:-1], plot_data['state_ref'][:, i+nq], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
          ax_x[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
          ax_x[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_x[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
          ax_x[i,1].grid(True)
          # Set xlim if any
          if(AUTOSCALE):
            ax_x[i,1].set_ylim(-plot_data['pin_model'].velocityLimit[i], plot_data['pin_model'].velocityLimit[i]) 

          # Add xlabel on bottom plot of each column
          if(i == nq-1):
              ax_x[i,0].set_xlabel('t(s)', fontsize=16)
              ax_x[i,1].set_xlabel('t(s)', fontsize=16)
          # Legend
          handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
          fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

      # y axis labels
      fig_x.text(0.05, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=16)
      fig_x.text(0.49, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
      fig_x.subplots_adjust(wspace=0.27)
      # Titles
      fig_x.suptitle('State = joint positions, velocities', size=18)
      # Save fig
      if(SAVE):
          figs = {'x': fig_x}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_x

  def plot_mpc_control(self, plot_data, PLOT_PREDICTIONS=False, 
                              pred_plot_sampling=100, 
                              SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                              SHOW=True, AUTOSCALE=False):
      '''
      Plot control data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting control data...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      dt_simu = plot_data['dt_simu']
      nq = plot_data['nq']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot-dt_simu, N_simu)
      t_span_plan = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
      # For each joint
      for i in range(nq):

          if(PLOT_PREDICTIONS):

              # Extract state predictions of i^th joint
              u_pred_i = plot_data['u_pred'][:,:,i]

              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_u_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts'])-1)]) #np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                  # Set up lists of (x,y) points for predicted positions and velocities
                  points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
                  lc_u.set_array(tspan_u_pred)
                  # Customize
                  lc_u.set_linestyle('-')
                  lc_u.set_linewidth(1)
                  # Plot collections
                  ax_u[i].add_collection(lc_u)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                  my_colors = cm(colors)
                  ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

          # Joint torques
          ax_u[i].plot(t_span_plan, plot_data['u_pred'][:,0,i], color='b', marker='.', linestyle='-', label='Optimal control u0*', alpha=0.6)
        #   ax_u[i].plot(t_span_plan, plot_data['u_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Predicted (PLAN)', alpha=0.1)
          ax_u[i].plot(t_span_simu, plot_data['u_mea'][:, i], color='r', linestyle='-', marker='.', label='Measured', alpha=0.1)
        #   ax_u[i].plot(t_span_simu, plot_data['grav'][:-1,i], color=[0.,1.,0.,0.], marker=None, linestyle='-.', label='Reg', alpha=0.9)
          # Plot reference
          if('ctrlReg' or 'ctrlRegGrav' in plot_data['WHICH_COSTS']):
              ax_u[i].plot(t_span_plan, plot_data['ctrl_ref'][:, i], linestyle='-.', color='k', marker=None, label='uReg_ref', alpha=0.5)
          ax_u[i].set_ylabel('$u_{}$'.format(i), fontsize=12)
          ax_u[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_u[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_u[i].grid(True)
          if(AUTOSCALE):
            ax_u[i].set_ylim(-plot_data['pin_model'].effortLimit[i], plot_data['pin_model'].effortLimit[i]) 

          # Last x axis label
          if(i == nq-1):
              ax_u[i].set_xlabel('t (s)', fontsize=16)
          # LEgend
          handles_u, labels_u = ax_u[i].get_legend_handles_labels()
          fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
      # Sup-y label
      fig_u.text(0.04, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=16)
      # Titles
      fig_u.suptitle('Control = joint torques', size=18)
      # Save figs
      if(SAVE):
          figs = {'u': fig_u}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 

      return fig_u

  def plot_mpc_ricatti_diag(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
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
      logger.info('Plotting Ricatti diagonal...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_K, ax_K = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
      # For each joint
      for i in range(nq):
          # Diagonal terms
          ax_K[i,0].plot(t_span_plan_u, plot_data['Kp_diag'][:, 0, i], 'b-', label='Diag of Ricatti (Kp)')
          ax_K[i,0].set_ylabel('$Kp_{}$'.format(i)+"$_{}$".format(i), fontsize=12)
          ax_K[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_K[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_K[i,0].grid(True)
          # Diagonal terms
          ax_K[i,1].plot(t_span_plan_u, plot_data['Kv_diag'][:, 0, i], 'b-', label='Diag of Ricatti (Kv)')
          ax_K[i,1].set_ylabel('$Kv_{}$'.format(i)+"$_{}$".format(i), fontsize=12)
          ax_K[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_K[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_K[i,1].grid(True)
          if(i == nq-1):
              ax_K[i,0].set_xlabel('t (s)', fontsize=16)
              ax_K[i,1].set_xlabel('t (s)', fontsize=16)
      # y axis labels
      fig_K.text(0.05, 0.5, '$K_p$', va='center', rotation='vertical', fontsize=16)
      fig_K.text(0.48, 0.5, '$K_v', va='center', rotation='vertical', fontsize=16)
      fig_K.subplots_adjust(wspace=0.27)  
      # Titles
      fig_K.suptitle('Diagonal Ricatti feedback gains K', size=16)
      # Save figs
      if(SAVE):
          figs = {'K_diag': fig_K}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_K

  def plot_mpc_Vxx_eig(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
      '''
      Plot Vxx eigenvalues
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting Vxx eigenvalues...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
      # For each state
      for i in range(nq):
          # Vxx eigenvals
          ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_eig'][:, 0, i], 'b-', label='Vxx eigenvalue')
          ax_V[i,0].set_ylabel('$\lambda_{}$'.format(i), fontsize=12)
          ax_V[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_V[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_V[i,0].grid(True)
          # Vxx eigenvals
          ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_eig'][:, 0, nq+i], 'b-', label='Vxx eigenvalue')
          ax_V[i,1].set_ylabel('$\lambda_{%s}$'%str(nq+i), fontsize=12)
          ax_V[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_V[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_V[i,1].grid(True)
          # Set xlabel on bottom plot
          if(i == nq-1):
              ax_V[i,0].set_xlabel('t (s)', fontsize=16)
              ax_V[i,1].set_xlabel('t (s)', fontsize=16)
      # # y axis labels
      # fig_V.text(0.05, 0.5, 'Eigenvalue', va='center', rotation='vertical', fontsize=16)
      # fig_V.text(0.49, 0.5, 'Eigenvalue', va='center', rotation='vertical', fontsize=16)
      # fig_V.subplots_adjust(wspace=0.27)  
      # Titles
      fig_V.suptitle('Eigenvalues of Value Function Hessian Vxx', size=16)
      # Save figs
      if(SAVE):
          figs = {'V_eig': fig_V}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_V

  def plot_mpc_Vxx_diag(self, plot_data, SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
      '''
      Plot Vxx diagonal terms
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
      '''
      logger.info('Plotting Vxx diagonal...')
      T_tot = plot_data['T_tot']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      nq = plot_data['nq']

      # Create time spans for X and U + Create figs and subplots
      t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
      fig_V, ax_V = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
      # For each state
      for i in range(nq):
          # Vxx diag
          ax_V[i,0].plot(t_span_plan_u, plot_data['Vxx_diag'][:, 0, i], 'b-', label='Vxx diagonal')
          ax_V[i,0].set_ylabel('$Vxx_{}$'.format(i), fontsize=12)
          ax_V[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_V[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_V[i,0].grid(True)
          # Vxx diag
          ax_V[i,1].plot(t_span_plan_u, plot_data['Vxx_diag'][:, 0, nq+i], 'b-', label='Vxx diagonal')
          ax_V[i,1].set_ylabel('$Vxx_{%s}$'%str(nq+i), fontsize=12)
          ax_V[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax_V[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax_V[i,1].grid(True)
          # Set xlabel on bottom plot
          if(i == nq-1):
              ax_V[i,0].set_xlabel('t (s)', fontsize=16)
              ax_V[i,1].set_xlabel('t (s)', fontsize=16)
      # y axis labels
      # fig_V.text(0.05, 0.5, 'Diagonal Vxx', va='center', rotation='vertical', fontsize=16)
      # fig_V.text(0.49, 0.5, 'Diagonal Vxx', va='center', rotation='vertical', fontsize=16)
      # fig_V.subplots_adjust(wspace=0.27)  
      # Titles
      fig_V.suptitle('Diagonal of Value Function Hessian Vxx', size=16)
      # Save figs
      if(SAVE):
          figs = {'V_diag': fig_V}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig_V
