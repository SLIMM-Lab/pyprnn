#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:40:00 2023

@author: malvesmaia
"""

import random
import numpy as np
import os
from plottingTool import helper
from plottingPRNN import plotPRNN
import torch

rseed = 13
random.seed ( rseed )

# ----------------------------------------------------------------------
# Auxiliary functions
# ----------------------------------------------------------------------

torch.manual_seed(rseed)
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def readSmpFile(infilename):
        
    infile = open(infilename, "r");
    tdata = infile.readlines()
    infile.close()
    
    sizetdata = len(tdata)
         
    for i in range(sizetdata):
        tdata[i] = tdata[i].split()
                        
    tdata = [x for x in tdata if x != []]
   
    sizetdata = len(tdata)
    
    print("# of samples: ", sizetdata)  
           
    y = np.array([np.array(xi) for xi in tdata])
    
    smpplt = y.copy()
    smpplt = np.array(smpplt, dtype=np.float32)
                  
    return smpplt

# ----------------------------------------------------------------------
# Input data 
# ----------------------------------------------------------------------

dim   = 3          # number of strains components
ls1 = 0            # size of nonlinear layer before material layer
bulkPts = 6        # number of bulk points 
ls2 = bulkPts*3          # size of material layer
which_network = 'jemjive'           # network was trained in python or jemjive?

# if the weights are coming from jemjive (.net), then the file 
# mnnfilelist will be read
# if the weights are coming from the python code (.pth), then the file
# pythonFile will be read

# Files with optimal weights 

homedir = '/home/malvesmaia/ownCloud/PhDMaterial/2024/Ehsan/code/'
mnnfilelist = ['mnn.net']
pythonFile = ['prnnloadnonprop_84442_18_36curves1layer.pth']

# File with test/validation/training set for plotting
    
lc01 = 'valset_060.data'

# If there is a file with the predictions of the test/validation/training set,
# set offline to True and provide the name of the file (typically 
# 'mnnoffline' + some naming for the type of loading, but you can change that
# below as required). Else, set offline as False and a network created in python
# will do the rest based on the weights provided (either from a .pth or .net 
# set of weights/biases)

offline = False
which_level = 'level3'

# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------

# This was once used for plotting stuff with other types of networks
# for now, plotting only with one kind ( MNN/PRNN ) is enough

which_nn = [True, False, False]   # MNN, RNN and BNN in this order
plot_curves = True                # Plot strain x stress for each curve
plot_which_curves = [True, False, False]
plot_nn = True               # Plot neural network state ( material points )
eval_error = True            # Evaluate error between NN and HF ( entire set )
eval_error_each = True       # Same as above, but gives error for each curve
plot_steps = False           # Time step x stress kind of plot

# Automatic

offline_name = mnnfilelist[0][:-7] + 'mnnoffline' + which_level + '.out'
testfilelist = [lc01] 

nneural = 0
for usenn in which_nn:
    if usenn == True:
        nneural += 1
        
errmnn = np.array([])

# ----------------------------------------------------------------------
# Running a loop with different test files (if necessary)
# ----------------------------------------------------------------------

for testfile in testfilelist:

    nsteps = int(testfile[:-5][-3:])
    combinedmnndisp = np.array([])
    
    for cont in range(len(mnnfilelist)):   
        stressptsnn = []
        strainptsnn = []
        histptsnn = [] 
        sizesnn = []
               
        print('========= Loading case ', testfile, ' ==============')
        printer = helper(testfile, homedir)
        combinedhf = printer.readCombinedFile ( nsteps, testfile, dim*2 )
                                                                    
        if (which_nn[0]):                 
            # Getting data from test/val/training set file
            
            valdataset = readSmpFile(testfile)
            
            # Get number of load cases 

            global nlcval 
            nlcval = int(len(valdataset)/nsteps)
            valsizes = [nsteps] * nlcval               
            nPointsVal = len(valdataset)

            print('# load cases: ', nlcval)
            print('# length data: ', valdataset.shape)
            
            # Prepare data 

            valstrain = valdataset[:, 0:3]
            valsig = valdataset[:, 3:6]
            
            # Clean start in case offline file is not available
            # This will create a NN with the architecture specified in 
            # the annmodel.py and number of points provided here
            
            if not offline:
                addressMNNfile = homedir
                combinedFileAddress = os.path.join ( addressMNNfile, 'combinedmnndisp.data')
                os.system ('rm ' + combinedFileAddress)
                
                mnn = plotPRNN ( ls1, ls2, bulkPts, which_network,
                           mnnfilelist[cont], pythonFile[cont] )
            
            init = 0
            sVal = nsteps            
                        
            for i in range(nlcval):
                
                 # Create network in python in case file with the predictions is
                 # not provided 

                 if not offline:
                    strain = valstrain[init:sVal, :]
                    init = sVal
                    sVal += nsteps
                    
                    print('Processing loading path %d/%d' % (i, nlcval) )
                                        
                    if (plot_nn):
                        # In this case, more info is retrieved
                        
                        stress, strainpts, strpts, histpts = mnn.saveNetwork( strain ) 
                       
                        stressptsnn = np.append(stressptsnn, strpts)
                        strainptsnn = np.append(strainptsnn, strainpts)
                        histptsnn = np.append(histptsnn, histpts)
                        stress = stress.cpu().reshape([nsteps, 3]) 
                        sizesnn = np.append(sizesnn, len(stress))
                    else:
                        stress = mnn.evalNetwork ( strain )
                                
                    # Writing an output file with the predictions     
                        
                    for m in range(len(strain)):
                        if (m == 0):
                            skipline = np.array(['\n'])
                        else:
                            skipline = np.append(skipline, ['\n'])
                    
                    smp = np.column_stack([strain, stress, skipline])                    
                    smp = smp.flatten()         
                    
                    newfilename = homedir + 'samplenndisp' + str(i) + '.data'
                    predfile = open(newfilename, 'w')                 
                    for j in smp:
                       predfile.write(str(j) + " ")              
                    predfile.close()
                    
                    # Thinned data and adding everything to one file
                    
                    nlines = len(stress)
                    idxs   = np.round( np.linspace (0, nlines - 1, nsteps ) ).astype(int)
                                       
                    thinnedfname = homedir + 'thinned.data'
                    with open ( newfilename, 'r' ) as data, open ( thinnedfname, 'w' ) as thinned:
                        for i, step in enumerate ( data,start=0 ):
                            if i in idxs:
                                thinned.write ( step )
                        thinned.write('\n')
                    os.system ('cat ' + thinnedfname + ' >> ' + combinedFileAddress)  
                    
                    # Remove individual file
                    
                    os.system('rm ' + newfilename)
                 else:
                    pass
            
            # Evaluating the error  
            
            if ( eval_error ): 
                if not offline:
                    combinedmnndisp = printer.readCombinedFile ( nsteps, combinedFileAddress)
                    aux = combinedmnndisp
                else:
                    offline_name =  mnnfilelist[cont][:-7] + 'mnnoffline' + which_level + '.out' 
                    aux = printer.readCombinedFile ( nsteps, offline_name, dim*2 )
                    combinedmnndisp = np.append(aux, [combinedmnndisp])
                eabs, erel = printer.evalError(combinedhf, aux, nlcval, nsteps, dim)
                errmnn = np.append(errmnn, [eabs, erel])
                
            if ( eval_error_each ):
                for ilc in range ( nlcval ):
                    errorilc = printer.evalError ( combinedhf[ilc*nsteps:(ilc+1)*nsteps, :], aux[ilc*nsteps:(ilc+1)*nsteps, :], 1, nsteps, dim )
                    print('Loading path %d error: %.2f MPa' % (ilc, errorilc[0]))
            
            # Reshaping stuff
            
            if offline:
                combinedmnndisp = combinedmnndisp.reshape([int((combinedmnndisp.shape[0])/(dim*2)), dim*2])
            else:
                combinedmnndisp = combinedmnndisp.reshape([int(combinedmnndisp.shape[0]), dim*2])
            
    # Plotting curves and neural network
                 
    if ( plot_curves ):
         printer.plotCurves(combinedhf, combinedmnndisp, combinedmnndisp,
                            combinedmnndisp, nsteps, plot_which_curves, dim, False)

    if ( plot_nn ):
        # Last argument works for plotting the time step vs stress view
        # instead of the usual strain/stress
        
        printer.plotNetwork(strainptsnn, stressptsnn, histptsnn, sizesnn, ls2,
                            bulkPts, combinedhf, combinedmnndisp, True )
      
print('\n------- Summary -------')
if ( which_nn[0] ):
    errmnn = np.reshape(errmnn, [len(testfilelist),  len(mnnfilelist), 2])
    print('Abs. error of entire set ', errmnn[0])
