#!/usr/bin/env python
"""Calculate the values of nitrate concentration (c) in time and space following the advection-diffusion equations.

Example usage from the notebook::

import adv_dif_1D_arr
# Run 5 time steps on a 9 point grid
adv_dif_1D_arr.adv_dif_1D(5,9)

Example usage from the shell::

  # Run 5 time steps on a 9 point grid
  $ adv_dif_1D_arr.py 5 9

The graph window will close as soon as the animation finishes.  And
the default run for 5 time steps doesn't produce much of interest; try
at least 100 steps.

Example usage from the Python interpreter::

  $ python
  ...
  >>> import adv_dif_1D_arr
  >>> # Run 200 time steps on a 9 point grid
  >>> adv_dif_1D_arr.adv_dif_1D((200, 9))
"""
from __future__ import division
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as colorbar
import os,glob

def init(x,n_grid,n_time,L,D,dt):
    """Set the initial condition values."""
    c_vals = np.zeros((n_time+1,n_grid+4))
    #c_vals[0, 2:n_grid+2]= np.sin((np.pi*x)/(2*L))
    b = x[len(x) // 2]
    t_0 = dt
    c_vals[0, :] = (1/np.sqrt(4*np.pi*D*t_0))*np.exp(-((x-b)**2)/(4*D*t_0))
    return c_vals

def boundary_conditions(c_array, n_grid):
    """Set the boundary condition values."""
    #c_array[0] = 0
    #c_array[1] = 0
    #c_array[2] = 0
    #c_array[3] = 0
    #c_array[n_grid] = c_array[n_grid - 1]
    #c_array[n_grid + 1] = c_array[n_grid]
    #c_array[n_grid + 2] = c_array[n_grid + 1]
    #c_array[n_grid + 3] = c_array[n_grid + 2]

    c_array[0] = c_array[n_grid]
    c_array[1] = c_array[n_grid+1]
    c_array[n_grid+2] = c_array[2]
    c_array[n_grid+3] = c_array[3]
    return c_array


def fw_euler(ts, c_now, D, dt, dx, n_grid):
    """Calculate the next time step values using the forward Euler scheme"""
    c_df = np.zeros(n_grid+4)
    for pt in np.arange(3, n_grid + 2):
        c_df[pt] = (D*dt/(dx**2)) * (c_now[pt + 1] - 2*c_now[pt] + c_now[pt - 1]) 
        
    # NOTE: This is the forward Euler calculation w/ current concentration subtracted to get only the diffusive contribution, and not the concentration at the next time step
        
    return c_df

def gettable(order, Numpoints):

    '''read in the corresponding coefficient table for the calculation of coefficients for advection3
    '''

    # create a matrix to store the table to be read in
    temp = np.zeros(5)
    ltable = np.zeros((order + 1, 5))

    fname = 'Tables/l{0}_table.txt'.format(order)
    fp = open(fname, 'r')

    for i in range(order+1):
        line = fp.readline()
        temp = line.split()
        ltable[i, :]= temp

    fp.close()
    return ltable

def bott(ts, c_now, order, Numpoints, u, dt, dx, epsilon): # IMPORTANT: Function taken and modified from Lab 10
    '''Step algorithm for Bott Scheme'''
    c_adv = np.zeros(Numpoints+4)
    ltable = gettable(order, Numpoints)

    # create a matrix to store the current coefficients a(j, k)
    amatrix = np.zeros((order+1, Numpoints))
    cmatrix = np.zeros((1, Numpoints+4))
    cmatrix[0,:] = c_now

    for base in range(0,5):
        amatrix[0:order+1, 0:Numpoints] += np.dot(
            ltable[0:order+2, base:base+1],
            cmatrix[:,0+base:Numpoints+base])

    # calculate I of c at j+1/2 , as well as I at j
    # as these values will be needed to calculate i at j+1/2 , as
    # well as i at j

    # calculate I of c at j+1/2(Iplus),
    # and at j(Iatj)
    Iplus = np.zeros(Numpoints)
    Iatj = np.zeros(Numpoints)

    tempvalue= 1 - 2*u*dt/dx
    for k in range(order+1):
        Iplus += amatrix[k] * (1- (tempvalue**(k+1)))/(k+1)/(2**(k+1))
        Iatj += amatrix[k] * ((-1)**k+1)/(k+1)/(2**(k+1))
    Iplus[Iplus < 0] = 0
    Iatj = np.maximum(Iatj, Iplus + epsilon)

    # finally, calculate the advective contribution to the concentration
    c_adv[3:Numpoints+2] = (
        c_now[3:Numpoints+2] *
        (-Iplus[1:Numpoints]/ Iatj[1:Numpoints]) +
        c_now[2:Numpoints+1]*
        Iplus[0:Numpoints-1]/ Iatj[0:Numpoints-1]) 
    # NOTE: This calculation differs from the equation in Lab 10. The current concentration has been subtracted to get only the advective contribution, and not the concentration at the next time-step
    
    return c_adv

def adv_dif_1D(args):
    """Run the model.

    args is a 4-tuple; (number-of-time-steps, number-of-grid-points, L, T)
    """
    n_grid = int(args[0]) # Number of length-steps
    n_time = int(args[1]) # Number of time-steps
    L = int(args[2]) # Total domain length [m]
    T = int(args[3]) # Total tme [s]

    # Constants and parameters of the model
    u = 0.1                     # flow field [m/s]
    dx = L / (n_grid+3)        # grid spacing [m]
    dt = T / n_time        # time step [s]
    c0 = 0.1                   # initial concentration [kg/L]
    D = 0.01                    # Diffusivity [m^2/s]
    x = np.arange(0,L+dx,dx) - 2*dx # length coordinate [m]
    t = np.arange(0,T+dt,dt) # time coordinate [s]
    epsilon = 0.0001
    order = 4
    
    # Initialize the concentration array
    c_vals_ad = init(x,n_grid,n_time,L,D,dt)
    c_vals_df = init(x,n_grid,n_time,L,D,dt)
    
    c_an_ad = init(x,n_grid,n_time,L,D,dt) # Analytical solution for advection
    c_an_df = init(x,n_grid,n_time,L,D,dt) # Analytical solution for diffusion
    
    # Impose boundary conditions on initial concentration and set initial concentration as current time-step
    c_now_ad = c_vals_ad[0,:]
    c_now_ad = boundary_conditions(c_now_ad,n_grid)
    c_vals_ad[0,:] = c_now_ad
    
    c_now_df = c_vals_df[0,:]
    c_now_df = boundary_conditions(c_now_df,n_grid)
    c_vals_df[0,:] = c_now_df
    
    # Loop over time domain
    for ts in np.arange(0, n_time):
        # Calculate the advective and diffusive contributions to the concentration at the current time-step
        adv = bott(ts, c_now_ad, order, n_grid, u, dt, dx, epsilon) # Calculate the advection contribution to the current concentration at this time-step
        df = fw_euler(ts, c_now_df, D, dt, dx, n_grid) # Calculate the diffusion contribution to the current concentration at this time-step
        c_next_ad = c_now_ad + adv # Add the advective and diffusive components to the current concentration to get the concentration at the next time-step
        c_next_df = c_now_df + df
        
        # set the boundary points
        c_next_ad = boundary_conditions(c_next_ad, n_grid)
        c_next_df = boundary_conditions(c_next_df, n_grid)
        
        c_vals_ad[ts+1,:] = c_next_ad # Store the new concentration
        c_now_ad = c_next_ad # Advance in time by one time-step

        c_vals_df[ts+1,:] = c_next_df
        c_now_df = c_next_df
        
        c_an_ad[ts,:] = analytical_adv(L,x,u,D,t[ts]+dt)
        c_an_df[ts,:] = analytical_diff(L,x,u,D,t[ts]+dt)
    return c_vals_ad, c_vals_df, c_an_ad, c_an_df

def analytical_adv(L,x,u,D,t):
    """Find the anlytical solution at each time-step for just advection for initial condition c(x,o)=sin(pi*x/2L).

    (L,x,u) should be same as in adv_dif_1D, t should be time input for each time-step
    """
    #c_ad = np.sin((np.pi*(x-u*t)) / (2*L))
    b = x[len(x) // 2]
    t_0 = 1
    c_ad = (1/np.sqrt(4*np.pi*D*t_0))*np.exp(-((x-u*t-b)**2)/(4*D*t_0))
    return c_ad

def analytical_diff(L,x,u,D,t):
    """Find the anlytical solution at each time-step for just diffusion for initial condition c(x,o)=sin(pi*x/2L).

    (L,x,D) should be same as in adv_dif_1D, t should be time input for each time-step
    """
    #c_df = np.sin((np.pi*x) / (2*L))*np.exp(-D*t*(np.pi/(2*L))**2)
    b = x[len(x) // 2]
    c_df = (1/np.sqrt(4*np.pi*D*t))*np.exp(-((x-b)**2)/(4*D*t))
    return c_df
    