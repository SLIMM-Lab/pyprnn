#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:40:00 2023
Modified on Wed Jun 12 14:42:00 2024            
@author: malvesmaia
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator

class PlotNN ( ):
    def __init__(self, testloader, models, labels = [], curveId = 1, trloader = None):    
        # testloader: Test dataloader
        # models: list of models
        # labels: list of labels for each model (optional)
        # curveId: index for plotting specific curve from testloader (optional) 
        # trloader: Training dataloader (optional)

        # Models used for plotting
        
        self.nnlist = models
        self.labels = labels
        self.nModels = len(self.nnlist)
        
        if len(labels) < self.nModels:
            print("\nWARNING: Number of labels does not match the number of"
                   " models. Using automatic labels instead.")
            self.labels = []
            for i in range(self.nModels): 
                self.labels = np.append(self.labels, ['NN ' + str(i)])
            

        # Test/validation/training set for plotting
        self.testLoader = testloader        
        self.current_id = curveId-1                 # Python starts at 0
        self.nlc = len(self.testLoader.dataset)     # Number of load cases (test)
        self.dim   = 3                              # Number of strains components
        self.nsteps = self.testLoader.dataset[0][0].shape[0]
 
        print('# of models: ', self.nModels)
        print('# test load paths: ', self.nlc)
        print('# time steps: ', self.nsteps)
        
        self.plot3D = False
        if not trloader == None:
            self.trainingLoader = trloader
            self.nstepsTr = self.trainingLoader.dataset[0][0].shape[0]
            self.nlctr = len(self.trainingLoader.dataset) # Number of load cases (training)
            self.plot3D = True
            print('# training load paths: ', self.nlctr)
            print('# time steps: ', self.nstepsTr)
        
        # Define settings for the buttons
        self.buttons = {}
        self.button_positions = {}

        self.defaults = { 'next':{
            'index':1,
            'hovercolor':'0.975',
            'label':'Next',
            'update':'next',
            'position':1
        },
          'previous':{
            'index':1,
            'hovercolor':'0.975',
            'label':'Previous',
            'update':'previous',
            'position':0
        },
            'random':{
            'index':2,
            'hovercolor':'0.975',
            'label':'Random',
            'update':'random',
            'position':2
        }}

        self.settings = {}
  	    
        # Initialize plot
        if 'height' in self.settings:
            self.h = self.settings['height']

        if 'width' in self.settings:
            self.w = self.settings['width']
            
        # Define color scheme
        self.set_colors()

        # Get the given axes from the settings, or create a new figure
        if 'ax' in self.settings:
            self.axs = self.settings['axs']
            self.fig = self.axs.figure
        else:
            self.fig = plt.figure(figsize = (10,8))
            gs = GridSpec(3, 11)     # Divide figure in a grid-like figure
            if self.plot3D:
                self.axs = [self.fig.add_subplot(gs[0, 0:5]),
                            self.fig.add_subplot(gs[1, 0:5]),
                            self.fig.add_subplot(gs[2, 0:5]),
                            self.fig.add_subplot(gs[0:2,6:10], projection='3d')  ] 
            else:
                self.axs = [self.fig.add_subplot(gs[0, 0:5]),
                            self.fig.add_subplot(gs[1, 0:5]),
                            self.fig.add_subplot(gs[2, 0:5])]                
            self.settings = {'axs': self.axs}

        # Initialize plots and labels
        self.plot_micro = [self.axs[0].plot([], []), self.axs[1].plot([], []), 
                           self.axs[2].plot([], [])]        
        self.plot_prnn = []
        self.plot_highlight = []        
        self.xlabels = [r'$\varepsilon_{xx}$ [-]', r'$\varepsilon_{yy}$ [-]', r'$\varepsilon_{xy}$ [-]']
        self.ylabels = [r'$\sigma_{xx}$ [MPa]', r'$\sigma_{yy}$ [MPa]', r'$\sigma_{xy}$ [MPa]']        
        self.palette = [self.c05, self.c01, self.c09]

        self.shown = False
        self.show()
        
    def set_colors(self):
        maxc = 255.0
        self.c01 = [ 141/maxc, 211/maxc, 199/maxc]
        self.c02 =  [ 255/maxc, 255/maxc, 179/maxc ]
        self.c03 =  [ 190.0/maxc, 186.0/maxc, 218/maxc  ]
        self.c04 = [ 251.0/maxc, 128.0/maxc, 114/maxc  ]
        self.c05 = [ 128.0/maxc, 177.0/maxc, 211/maxc  ]
        self.c06 =  [ 128.0/maxc, 177.0/maxc, 211/maxc  ]
        self.c07 = [ 253.0/maxc, 180.0/maxc, 98/maxc  ]
        self.c08 =  [ 179.0/maxc, 222.0/maxc, 105/maxc  ]
        self.c09 =  [ 252.0/maxc, 205.0/maxc, 229/maxc  ]
        self.c10 =  [ 217.0/maxc, 217.0/maxc, 217/maxc  ]
        self.c11 =  [ 102.0/maxc, 102.0/maxc, 102/maxc  ]
        self.c12 =  [ 90.0/maxc, 174.0/maxc, 159/maxc  ]
        self.c13 = [ 33.0/maxc, 201.0/maxc, 160/maxc  ]
       
    def add_buttons(self, *var_list, **settings):       
        for var in var_list:
            self.add_button(var, **settings)

    # Add a button to the bottom or left side of the plot
    def add_button(self, var, **settings):

        # Check if the variable is in defaults
        def_settings = self.defaults.get(var, {})

        # Load all default/given values
        hovercolor = settings['hovercolor'] if 'hovercolor' in settings else def_settings['hovercolor']
        label = settings['label'] if 'label' in settings else def_settings['label']
        update = settings['update'] if 'update' in settings else def_settings['update']
        position = settings['position'] if 'position' in settings else def_settings['position']

        # Create the button
        # Note: it is important that the button is not created in exactly the same place as before
        # otherwise, matplotlib will reuse the same axis
        
        ax_button = self.fig.add_axes([0.26+0.07 * len(self.buttons), .89, 0.07, 0.04])
        button = Button(
            ax=ax_button, 
            label=label,
            hovercolor=hovercolor
        )
        button.label.set_fontsize(12)

        # Add the button to the dictionary that will store the button values
        self.buttons[var] = button
        self.button_positions[var] = position

        # Get the correct update function
        update_func = self.get_update_func(update)

        # Add an event 
        button.on_clicked(update_func)

    # This function takes a string and returns the corresponding update function
    def get_update_func(self, update):
        if update == 'next':
            return self.plot_pred_next
        elif update =='previous':
            return self.plot_pred_previous
        if update == 'random':
            return self.plot_pred_random
        else:
            print('Else')
            return super().get_update_func(update)

    # Define a show function, so importing matplotlib is not strictly 
    # necessary in the notebooks
    def show(self):
        # Update the plot
        self.eval_pred ( self.current_id )

        # Check if show has already been called
        if 'ax' in self.settings:
            self.shown = True
        # Check if the plot has already been shown
        if not self.shown:

            # If not, forward to plt.show()
            # Note that plt.show() should only be called once!
            plt.show()

            # Remember that the plot has now been shown
            self.shown = True
   
    # Main plotting function
    def plot_curves(self, hf, nn, curve_id = 0): 
      
      # Use latex to render labels
      hfont = {'fontname':'serif', 'fontsize':13}
        
      # Checking lower and upper bounds of specified curve idx
      # if not within right bounds pick random curve
      
      if curve_id < 0 or curve_id >= self.nlc:   
        # Plotting random curve from dataloader
        curve_id = np.random.randint(0, self.nlc)   
        self.current_id = curve_id
        print('\nATTENTION: Specified index is not available. '
              'Plotting curve %d instead.' % int(curve_id+1))
 
      # Extracting strain and stress from hf data
      hfstrain = hf[0]
      hfstress = hf[1]
      
      # Initialize list to track all nn predictions being plotted
      plot_prnn = []

      # Loop over strain stress components
      for j in range(self.dim):          
        maxYnn = -1e6
        minYnn = 1e6
        
        # Remove old lines
        if ( self.shown == True ): 
            for line in self.axs[j].lines:
                line.remove()

        # Plot high fidelity data
        self.plot_micro[j], = self.axs[j].plot(hfstrain[:,j], hfstress[:,j], 
                                               color = self.c11, 
                                               linewidth = 2.5, 
                                               label= "Micro", 
                                               linestyle = "dotted", 
                                               zorder = 1)           
       
        # Plot nn predictions
        for netIdx in range (self.nModels):
            nnstrain = nn[netIdx, :, 0:3]
            nnstress = nn[netIdx, :, 3:6]
            
            maxYnn = max(max(nnstress[:,j]), maxYnn)
            minYnn = min(min(nnstress[:,j]), minYnn)
            
            aux, = self.axs[j].plot(nnstrain[:, j], nnstress[:, j], 
                                    linewidth = 1.5, color = self.palette[netIdx],
                                label = self.labels[netIdx], linestyle = "solid")
            plot_prnn.append(aux)
        
        # At the end of the loop through components and nns, save all 
        # plotted lines (they should be removed when a new curve is shown next)
        if ( j == self.dim - 1 ): self.plot_prnn = plot_prnn
                    
        # Getting upper and lower bounds of the current curve
        maxY = max(maxYnn, max(hfstress[:,j]))
        minY = min(minYnn, min(hfstress[:,j]))
        maxX = max(hfstrain[:,j]).detach().numpy()
        minX = min(hfstrain[:,j]).detach().numpy()
        
        # Add labels and set x and y limits 
        buff = 1.2
        self.axs[j].set_ylabel(self.ylabels[j],**hfont)
        self.axs[j].set_xlabel(self.xlabels[j],**hfont)
        self.axs[j].set_ylim([minY*buff, maxY*buff])
        self.axs[j].set_xlim([minX*buff, maxX*buff])
        self.axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        self.axs[j].yaxis.tick_left()
        self.axs[j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        self.axs[j].xaxis.set_ticks_position('both')
        self.axs[j].yaxis.set_ticks_position('both')
        self.axs[j].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[j].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[j].yaxis.set_major_locator(MaxNLocator(5))
        self.axs[j].tick_params(which='major', length=5, color='lightgrey', 
                                width=.7, direction='in', labelsize = 14)
        self.axs[j].tick_params(which='minor', length=3, color='lightgrey', 
                                width=.7, direction='in', labelsize = 14)
        if (j == 0):  
            self.axs[j].legend(loc=(1.04,0),  
                               frameon=False, prop={'size': 12.0})    
          
      # Set spacing between subplots      
      plt.subplots_adjust(hspace = 0.4)
            
      # Add title 
      self.axs[0].set_title('Curve %d/%d' % (curve_id+1, self.nlc), 
                            loc = 'left') 
      
      # Plot training curves in the strain space (if plot3D is True)
      
      if ( self.plot3D ):
          print('Plot 3d sphere.')
          self.rmax = -1e6
          colorTraining = 'lightgrey'
          colorTest = self.c07
          
          print('Print 3D')
          # TODO: generalize to receive as many training set as there are models
          for path in range(self.nlctr):          
              strainPath = self.trainingLoader.dataset[path][0].detach().numpy()
              
              # Each strain component is a coord
              # Adding initial state (= 0) to the beggining of
              # each path
              xx, yy, zz = np.hsplit(strainPath, 3) 
             
              xxprev = np.array([0.0])
              xxprev = np.append(xxprev, xx[:-1])
           
              yyprev = np.array([0.0])
              yyprev = np.append(yyprev, yy[:-1])
              
              zzprev = np.array([0.0])
              zzprev = np.append(zzprev, zz[:-1])
                    
              # Tracking largest radius to plot a 
              # sphere around training paths
              
              r = np.amax([np.amax(abs(xx)), 
                             np.amax(abs(yy)), np.amax(abs(zz))])
              
              self.rmax = max ( r, self.rmax )   
                
              self.axs[3].scatter(0,0,0,color="darkgrey",s=1.5)                 
              self.axs[3].scatter(xx[self.nstepsTr-1],
                                  yy[self.nstepsTr-1],
                                  zz[self.nstepsTr-1],color=colorTraining,s=1.0)      
              for i,j,k, m, n, o in zip(xx,yy,zz, xxprev, yyprev, zzprev):
                 self.axs[3].plot3D([i[0],m],[j[0],n],[k[0],o], 
                                     color = colorTraining, linewidth = 0.4, 
                                     linestyle = 'solid')
                
          # Plotting sphere around training data
              
          phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
          x = self.rmax*np.sin(phi)*np.cos(theta)
          y = self.rmax*np.sin(phi)*np.sin(theta)
          z = self.rmax*np.cos(phi)
          
          # Plot surface only once
          if self.shown == False:
                self.axs[3].plot_surface(x, y, z,  rstride=1, 
                                         cstride=1, color='c', 
                                         alpha=0.02, linewidth=0)
          self.axs[3].set_xlim([-self.rmax,self.rmax])
          self.axs[3].set_ylim([-self.rmax,self.rmax])
          self.axs[3].set_zlim([-self.rmax,self.rmax])
              
          # Highlight test curve 
          strainPath = self.testLoader.dataset[curve_id][0].detach().numpy()
          xx, yy, zz = np.hsplit(strainPath, 3) 
          xxprev = np.array([0.0])
          xxprev = np.append(xxprev, xx[:-1])
         
          yyprev = np.array([0.0])
          yyprev = np.append(yyprev, yy[:-1])
            
          zzprev = np.array([0.0])
          zzprev = np.append(zzprev, zz[:-1])
          
          if ( self.shown == True ): 
              # Clean previous curve
              for line in self.plot_highlight:
                  line.remove()
              self.plot_highlight = []

          for i,j,k, m, n, o in zip(xx,yy,zz, xxprev, yyprev, zzprev):
               aux, = self.axs[3].plot3D([i[0],m],[j[0],n],[k[0],o], 
                                   color = colorTest, linewidth = 1, 
                                   linestyle = 'solid')    
               self.plot_highlight.append(aux)
            
          # Setting plotting in 3D  
          self.axs[3].set_box_aspect((1,1,1))
            
          self.axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          self.axs[3].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          self.axs[3].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          self.axs[3].xaxis.set_ticks_position('both')
          self.axs[3].yaxis.set_ticks_position('both')
          self.axs[3].tick_params(axis = 'both', labelsize = 8)
             
          self.axs[3].xaxis._axinfo["grid"].update(linestyle = '-', linewidth = 0.2)
          self.axs[3].yaxis._axinfo["grid"].update(linestyle = '-', linewidth = 0.2)
          self.axs[3].zaxis._axinfo["grid"].update(linestyle = '-', linewidth = 0.2)  
            
          self.axs[3].set_xlabel(r"$\varepsilon_x$ [-]", fontsize = 10)
          self.axs[3].set_ylabel(r"$\varepsilon_y$ [-]", fontsize = 10, rotation = 0 )
            
          self.axs[3].zaxis.set_rotate_label(False) 
            
          self.axs[3].set_zticks(np.linspace(-self.rmax, self.rmax, 5))
          self.axs[3].set_xticks(np.linspace(-self.rmax, self.rmax, 5))
          self.axs[3].set_yticks(np.linspace(-self.rmax, self.rmax, 5)) 
            
          self.axs[3].set_zlabel(r"$\varepsilon_{xy}$"
                        "\n"
                        r"[-]", fontsize = 10, rotation = 0)    
          self.axs[3].w_xaxis.pane.fill = False
          self.axs[3].w_yaxis.pane.fill = False
          self.axs[3].w_zaxis.pane.fill = False
          
          # Angle to view the 3d sphere
            
          self.axs[3].view_init(elev=20., azim=55)
          
       # Allow for automatic updating of the plot
      self.fig.canvas.draw_idle()
      
    def eval_pred(self, curve_id = 0):
        if curve_id >= 0: self.current_id = curve_id
            
        nnpreds = np.array([])  # Collects predictions from all models
                
        # Loop over n models for specified curve only
        for netIdx in range(self.nModels):   
            nn = self.nnlist[netIdx]
                        
            strain = self.testLoader.dataset[curve_id][0]
            stressnn = nn.forward ( torch.unsqueeze(strain, 0) )            
            stressnn = stressnn.reshape([self.nsteps, 3]).detach().numpy()
            nnpred = np.column_stack([strain.detach().numpy(), stressnn])            
            
            # Add prediction using model netIdx to variable with predictions
            # from all models
            nnpreds = np.append(nnpreds, [nnpred])
                             
        # Reshaping stuff        
        nnpreds = nnpreds.reshape([
            self.nModels, int(nnpreds.shape[0]/(self.dim*2*self.nModels)), 
            self.dim*2])

        # Get high fidelity from provided testloader
        hfdata = self.testLoader.dataset[curve_id]
        
        #  Call main plot function         
        self.plot_curves(hfdata, nnpreds, self.current_id)
                                               
    def plot_pred_next(self, event):
        if (self.current_id+1) < self.nlc: self.current_id += 1    
        self.eval_pred(self.current_id)        

    def plot_pred_previous(self, event):
        if (self.current_id-1) >= 0: self.current_id -= 1            
        self.eval_pred(self.current_id)        

    def plot_pred_random (self, event):
        self.current_id = np.random.randint(0, self.nlc)     
        self.eval_pred(self.current_id)        
