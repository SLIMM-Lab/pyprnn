#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:44:35 2023

@author: malvesmaia
"""
from torch import nn
import torch
import J2Tensor_vect as J2Tensor
from customlayers import softLayer

class neural_network(nn.Module):
    def __init__(self,n_features,output_length, bulk, dev):
        super(neural_network,self).__init__()

        self.device = dev
        self.bulkPts = bulk
        self.nIntPts = self.bulkPts
        self.nComp = 3
        self.ls = self.bulkPts*self.nComp
        self.hidden_size = self.ls
        self.n_layers = 1        
        self.in_size = n_features
        self.output_length = output_length
        self.outputMatPts = False
                
        print('Input size ', self.in_size)
        print('Material layer size ', self.hidden_size)
        print('No. of material points ', self.nIntPts)
        print('Output layer size ', self.output_length)
        
        # Defining layers 
        
        # Linear -> Regular dense layer
        # softLayer -> Dense layer with sotfplus on weights (customized)
        
        self.fc1 = nn.Linear(in_features=self.in_size,out_features=self.hidden_size, device = self.device, bias = False)
        self.fc2 = softLayer(in_features=self.hidden_size,out_features=self.output_length, device = self.device, bias = False)
 
    def getOutputMatPts ( self, x):
        
        # This function is here for printing purposes only
        
        # Equivalent to propagate 
        
        batch_size, seq_len, _ = x.size()
        
        output =  x.clone()
        out = torch.zeros([batch_size,seq_len, self.output_length]).to(self.device)
        
        # Create material models
        
        childb = J2Tensor.J2Material(self.device) 
        
        # Create and configure fic. integration points
        
        localstrains = torch.zeros([batch_size, seq_len, self.ls]) 
        localstresses = torch.zeros([batch_size, seq_len, self.ls]) 
        localhistory = torch.zeros([batch_size, seq_len, self.bulkPts*1]) 
        
        ip_pointsb = batch_size*self.bulkPts
        childb.configure( ip_pointsb )        
        
        # Process each curve at a time

        for t in range(seq_len):
            
           # Encoder ( dehomogenization )
           
           outputt = self.fc1(output[:,t,:])
           localstrains[:, t, :] = outputt
           
           # Evaluating bulk models
              
           outputt = childb.update(outputt.view(ip_pointsb,self.nComp))
           childb.commit()
           localhistory[:,t, ] = childb.getHistory()

          # Decoder ( homogenization )
                       
           localstresses[:, t, :]  =  outputt.view(batch_size, self.ls)
            
           outputt = self.fc2(outputt.view(batch_size, self.ls))
           out[:,t, :] = outputt.view(-1,self.output_length)
               

        output=out.to(self.device)
        return output, localstrains, localstresses, localhistory
    
    def forward(self,x):
        
        # Equivalent to propagate 
        
        batch_size, seq_len, _ = x.size()
        
        output =  x.clone()
        out = torch.zeros([batch_size,seq_len, self.output_length]).to(self.device)
        
        # Create material models
        
        childb = J2Tensor.J2Material(self.device) 
        
        # Create and configure integration points
        
        ip_pointsb = batch_size*self.bulkPts
        childb.configure( ip_pointsb )        
        
        # Process each curve at a time

        for t in range(seq_len):
          # Encoder ( dehomogenization )
               
          outputt = self.fc1(output[:, t,:])
               
          # Evaluating stress and updating internal variables
          # from all material points at the same time         
          
          outputt = childb.update(outputt.view(ip_pointsb,self.nComp))
          childb.commit( ) 

    
    	  # Decoder ( homogenization )
		           
          outputt = self.fc2(outputt.view(batch_size, self.ls))
          out[:, t, :] = outputt.view(-1,self.output_length)

        output = out.to(self.device)
        return output
