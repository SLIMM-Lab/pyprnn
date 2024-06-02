#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 19:21:42 2021

@author: malvesmaia
"""

import random
random.seed ( 13 )
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator, MultipleLocator
from matplotlib import cm, colors
from colours import PrintingStyle 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as image
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

class helper ( ):
 def __init__(self, reference, homedir = "/home/malvesmaia/code/mnnmattestes"):
     self.name = reference
     self.rmax = -1e10
     self.plotsphere = True
     self.maxel = 0
     self.homedir = homedir
     self.style = PrintingStyle()

 def readOutFile(self, infilename):
    infile = open(infilename, "r")
    tdata = infile.readlines()
    
    infile.close()
        
    sizetdata = len(tdata)
    
    for i in range(sizetdata):
        tdata[i] = tdata[i].split()
        
    tdata= [x for x in tdata if x != []] 
    
    sizetdata = len(tdata)
                
    tdata = np.array(tdata, dtype=np.float32)
                   
    return tdata

 def readSampleFile(self, infilename ):
    infile = open(infilename, "r")
    tdata = infile.readlines()
    
    infile.close()
        
    sizetdata = len(tdata)
    
    for i in range(sizetdata):
        tdata[i] = tdata[i].split()
        
    tdata= [x for x in tdata if x != []]
    
    sizetdata = len(tdata)
                
    tdata = np.array(tdata, dtype=np.float32)
    
    return tdata 
             
 def readCombinedFile(self, nstep, filename, ncols = 6):
    tdata = self.readOutFile(filename)
    combined = tdata.reshape([np.round(len(tdata/nstep)), ncols])    
    return combined  

 def plotCurves(self, combined, combinedmnn, combinedrnn, combinedbnn, 
                nstep, which_nn, ncomp = 3, angle = 0,
                globrespdata = None): 
     
  if ncomp == 3: 
      self.plotCurves2D(combined, combinedmnn, combinedrnn, combinedbnn, 
                nstep, which_nn )
  else:
      raise Exception ( 'Not available ')
      
 def plotCurves2D(self, combined, combinedmnn, combinedrnn, combinedbnn,
                nstep, which_nn ):     
  ncomp = 3     
  nplot = int(len(combined)/nstep)  
     
  # Plotting 
  
  hfont = {'fontname':'serif', 'fontsize':14}
  
     
  for i in range(nplot):
   if i <= 3:           # Plotting 3 curves max.     
     init = nstep*i;
     end = nstep*(i+1)
     
     fig, axs = plt.subplots(3, 1, figsize = (4,5))  

     plt.rc('text', usetex=True)
     plt.rc('text.latex', preamble = r"\usepackage{amsmath} \usepackage{amssymb}")   
     plt.rc('font',family = 'serif',  size=16)     
     
     
     for j in range(ncomp): 
       axs[j].plot(combined[init:end, j], combined[init:end, j+3], color = self.style.c11, linewidth = 2.5, 
                   label= "Micro curve " + str(i), linestyle = "dotted", zorder = 10)
       if ( which_nn[0] ):
           axs[j].plot(combinedmnn[init:end, j], combinedmnn[init:end, j+3], 
                           color = self.style.c05, linewidth = 1.5, 
                              label = r"PRNN", linestyle = "solid")
       if ( which_nn[1]):
           axs[j].plot(combinedrnn[init:end, j], combinedrnn[init:end, j+3],
                           color = self.style.c13, linewidth = 1.5, 
                       label= r"18 Type V curves, material layer with $\mathcal{D}^{\omega}_1$", linestyle = "solid")
       if ( which_nn[2]):
           axs[j].plot(combinedbnn[init:end, j], combinedbnn[init:end, j+3],
                            color = self.style.c07, linewidth = 1.25, 
                             label = r"$\|\mathfrak{D}_{\mathrm{PRNN}}\|$ = 18 curves", linestyle = "solid")
                    
       if ( j == 0):
           axs[j].set_ylabel(r'$\sigma_\textrm{xx}$ [MPa]',**hfont)
           axs[j].set_xlabel(r'$\varepsilon_\textrm{xx}$ [-]',**hfont)
       elif ( j == 1): 
           axs[j].set_xlabel(r'$\varepsilon_\textrm{yy}$ [-]',**hfont)
           axs[j].set_ylabel(r'$\sigma_\textrm{yy}$ [MPa]' ,**hfont)
       else:
           axs[j].set_xlabel(r'$\varepsilon_\textrm{xy}$ [-]',**hfont);
           axs[j].set_ylabel(r'$\sigma_\textrm{xy}$ [MPa]',**hfont);
       
       for ax in axs:
           ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
           ax.yaxis.tick_left()
           ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
           ax.xaxis.set_ticks_position('both')
           ax.yaxis.set_ticks_position('both')
           ax.xaxis.set_minor_locator(AutoMinorLocator())
           ax.yaxis.set_minor_locator(AutoMinorLocator())
           ax.yaxis.set_major_locator(MaxNLocator(5))
           ax.tick_params(which='major', length=5, color='lightgrey', width=.7, direction='in', labelsize = 14)
           ax.tick_params(which='minor', length=3, color='lightgrey', width=.7, direction='in', labelsize = 14)
           if ( ax == axs[-1] and j == 2):  
               # Plot legend only when all axes are ready (j == 2) and
               # for the last box (axs[-1])
               ax.legend(loc="lower right",  frameon=False, prop={'size': 12.0})    
           plt.subplots_adjust(hspace = 0.65)
       
     plt.subplots_adjust(wspace = 0.275)           
     
     if ( i == 28):   # Plot only curve 28 (change as needed!) 
      figname = "curve" + str(i) + ".pdf"
      plt.savefig(figname, format='pdf', bbox_inches='tight', dpi = 300)
       
     plt.show()
     plt.close()
          
 def plotNetwork (self, strainpts, stresspts, histpts, sizes, sl, 
                  bulkPts, combined, combinednndisp, strainView = True ):
    
    nplot = len(sizes) 
    nip = bulkPts
    
    tool = PrintingStyle()

    totalsize = 0
    for i in sizes:
       totalsize = int(totalsize + i) 
       
    strainpts = strainpts.reshape([totalsize, sl])
    stresspts = stresspts.reshape([totalsize, sl])
    histpts = histpts.reshape([totalsize, bulkPts*1])
    
    hfont = {'fontname':'serif', 'fontsize':14}
    bulkColor = tool.c12
       
    for i in range(nplot):
        nstep = sizes[i]
                     
        if (i == 0):
            init = 0
            end = int(init + nstep)
        else:
            init = end
            end = int(end + nstep)
           
        strain = strainpts[init:end, :]
        stress = stresspts[init:end, :]
        hist = histpts[init:end, :]
        
        nrows = nip + 1
        fig, axs = plt.subplots( nrows, 4, figsize = (8.,1.5*nrows) )
        fig.delaxes(axs[nrows-1][3])
        
        plt.subplots_adjust(hspace = .4)
        plt.subplots_adjust(wspace=0.3)

        plt.rcParams.update({"text.usetex": True})
        plt.rc('text.latex', preamble = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{stmaryrd}")           

        for j in range(nip):     
            if j < bulkPts:
                ncomponents = 3
                initIdx = j*3
                histIdx = j*1
                colorpt = bulkColor
                
            for k in range( ncomponents ): 
              if ( strainView ):
                  xaxis = strain[:, initIdx + k]
                  xaxismacro = combinednndisp[init:end, k]
              else:
                  xaxis = np.linspace(1, int(nstep), int(nstep))
                  xaxismacro = np.linspace(1, int(nstep), int(nstep))
                
              axs[j,k].plot(xaxis, stress[:, initIdx + k], color = colorpt, 
                             linewidth = 1.2,  label= "J2 model", 
                             linestyle = "solid")
                  
              if ( k == 0):
                  axs[j,k].set_ylabel('$m_' + str (j+1) + '$',**hfont)  
                                          
              if ( j == 0 ):
                axs[nrows-1,k].plot(xaxismacro,
                              combinednndisp[init:end, k+3], color = bulkColor, linewidth = 1.2, 
                                 label= "J2 model", linestyle = "solid", zorder = -1)                   
                axs[nrows-1,k].plot(xaxismacro, combinednndisp[init:end, k+3], 
                                    color =tool.c11, linewidth = 1.2, 
                                 label= "PRNNFE$^2$", linestyle = "solid")    
                axs[nrows-1,k].plot(xaxismacro, combined[init:end, k+3], color ="black", linewidth = 1.2, 
                         label= "FE$^2$", linestyle = "dashed", zorder=10)  

                
                if (k == 0):
                  axs[j,k].set_title(r'$\sigma_\textrm{xx}$ [MPa]',**hfont)
                  axs[nrows-1,0].set_ylabel('Macro', **hfont)                
                elif ( k ==1 ):
                  axs[j,k].set_title(r'$\sigma_\textrm{yy}$ [MPa]',**hfont)
                else:
                  axs[j,k].set_title(r'$\sigma_\textrm{xy}$ [MPa]',**hfont)
                axs[j,3].set_title(r'$\varepsilon_\textrm{eq}^p$ [-]', **hfont)
                
                                                        
            axs[j,3].plot(np.linspace(1, int(nstep), int(nstep)), hist[:, histIdx], 
                          color = colorpt, linewidth = 1.2, label= "NNFE$^2$", linestyle = "solid")
            
            plt.rc('text', usetex=True)
            
            if ( j == nip-1): 
                if strainView:
                    axs[j , 3].set_xlabel('Time step [-]',**hfont)
                else:
                    axs[j , 3].set_xlabel('Time step [-]',**hfont)
                
                      
        cont = 0
        for ax in axs.flat:
            if ( cont != 3): 
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.tick_left()
                if ( strainView ):
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                else:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.tick_params(which='major', length=5, color='lightgrey', width=0.5, direction='in', labelsize = 8)
                ax.tick_params(which='minor', length=3, color='lightgrey', width=0.5, direction='in', labelsize = 8)
                plt.legend(bbox_to_anchor=(1.25,1.), frameon=False, prop={'size': 12.0},
                           loc="upper left", borderaxespad=0)
                handles, labels = ax.get_legend_handles_labels()
            else:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                ax.yaxis.tick_right()
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_label_position("right")
                ax.tick_params(which='major', length=5, color='lightgrey', width=.5, direction='in', labelsize = 6)
                ax.tick_params(which='minor', length=3, color='lightgrey', width=.5, direction='in', labelsize = 6)
            cont += 1     
            if ( cont == 4):
                cont = 0

        plt.show()
     
 def evalError(self, combined, combinednndisp, nplot, nstep, ncomp = 3): 
    mse = np.array([])
    mserel = np.array([])
    msepercomp = np.array([])
    for i in range ( nplot ): 
        init = i*nstep
        end = init + nstep
        mse = np.append(mse, np.sqrt(((combined[init:end,ncomp:ncomp*2] - combinednndisp[init:end,ncomp:ncomp*2])**2).sum(axis=1))) 
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    return mse.mean(axis=0), 0
