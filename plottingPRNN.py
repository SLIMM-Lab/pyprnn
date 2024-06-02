#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:25:11 2021

@author: malvesmaia
"""
import numpy as np
import torch
from annmodel import neural_network

# ----------------------------- GPU related ----------------------------------
    
is_cuda = torch.cuda.is_available()
   
if is_cuda:
  device = torch.device("cpu")
  print("GPU is available")
else:
  device = torch.device("cpu")
  print("GPU not available, CPU used")
  

# Material Neural Network (fully connected)

class plotPRNN ( ):
   def __init__(self, nls, ls, bulkPts = 0, 
                which_network = 'jemjive', 
                mnnFile = 'none', pythonFile = 'none',):
       
     self.nls = nls
     self.sl = ls
     self.pythonFile = pythonFile
     self.which_network = which_network
     
     if ( bulkPts == 0 ): 
         self.bulkPts = round ( ls / 3 )
     else:
         self.bulkPts = bulkPts
     
     if which_network == 'jemjive':
         if ( mnnFile == 'none' ):
             raise Exception ( 'No .net file was provided for reading optimal parameters.') 
             
         print ( '\nATTENTION: Reading weights/biases from jemjive (.net)\n' )
         
         self.wl1, self.bl1, self.wl2, self.bl2, self.wl3 = self.readNetwork(mnnFile)
         
         # Creating a python network based on weights read from .net file
         # Pay attention so that the architecture you want to test is 
         # the same as the one in the annmodel
         
         self.model = neural_network(n_features=3,output_length=3, 
                                           bulk = self.bulkPts, 
                                           dev = device).to(device)
         
         # Setting the weights
         
         with torch.no_grad():
             self.model.fc1.weight = torch.nn.Parameter ( torch.from_numpy(self.wl2).double() )
             self.model.fc1.biases = torch.nn.Parameter ( torch.from_numpy(self.bl2).double() )
             self.model.fc2.weight = torch.nn.Parameter ( torch.from_numpy(self.wl3).double() )
             self.model.eval()

     else:
        if ( pythonFile == 'none' ):
             raise Exception ( 'No .pth file was provided for reading optimal parameters.')          
         
        print ( '\nATTENTION: Reading weights/biases from python file (.pth)\n' )
        
        # Creating and loading network from python .pth file
        
        self.model = neural_network(n_features=3,output_length=3, 
                                           bulk = self.bulkPts, 
                                           dev = device).to(device)
        self.model.load_state_dict(torch.load(self.pythonFile))
        self.model.eval()
  
   def readNetwork ( self, infilename ):
    infile = open(infilename, "r")
    tdata = infile.readlines()
    infile.close()
        
    sizetdata = len(tdata)
    
    for i in range(sizetdata):
        tdata[i] = tdata[i].split()
        
    tdata= [x for x in tdata if x != []];  
   
    sizetdata = len(tdata)             
    tdata = np.array(tdata, dtype=np.float32)
        
    init = 0
    end = 3*self.nls
    wl1 = np.array(tdata[init:end])
   
    init = end
    end = init + self.nls
    bl1 = np.array(tdata[init:end])
    
    if (self.nls <= 0):
        prev = 3
    else:
        prev = self.nls
    
    init = end
    end = init + prev*self.sl 
    wl2 = np.array(tdata[init:end])
    
    init = end
    end = init + 0 #self.sl
    bl2 = np.array(tdata[init:end])
 
    init = end
    end = sizetdata
    wl3 = np.array(tdata[init:end]) 
    
    if ( self.nls > 0):
        wl1 = wl1.reshape([prev, 3])
  
    wl2 = wl2.reshape([self.sl,prev])
    wl3 = wl3.reshape([3,self.sl])
    
    return wl1, bl1, wl2, bl2, wl3

   def evalNetwork ( self, strain ):
      nsteps = len(strain)
      stresses = np.zeros([nsteps,3])

      strain = torch.Tensor(strain)
      strain = strain[None,:,:]
      with torch.no_grad():
          stresses = self.model ( strain ).cpu().reshape([nsteps, 3]) 
          
      return stresses

   def saveNetwork ( self, strain ):
       
      strain = torch.Tensor(strain)
      strain = strain[None,:,:]
      
      with torch.no_grad():
         stresses, strainpts, stresspts, histpts = self.model.getOutputMatPts ( strain )
          
      return stresses, strainpts, stresspts, histpts       
