#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:25:38 2022

@author: malvesmaia
"""

# Adapted:       Marina A. Maia
# Date:          Dec 2021

# Adapted:       Frans P. van der Meer
# Date:          May 2024

# ------------------------ Importing libraries -------------------------------

import os
import torch
import copy
import time
import numpy as np
from numpy import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from annmodel import neural_network
import random

# ----------------------------- GPU related ----------------------------------

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cpu")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# ------------------------------ Dataset -------------------------------------

class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float64).to(device)
        self.y = torch.tensor(y, dtype=torch.float64).to(device)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# ------------------------ Optimization problem -----------------------------

class PRNN ():
    def __init__(self, trainingdata, bulkPoints, **kwargs ):

        self.n_features = 3
        self.trainingData = trainingdata
        self.bulkPts = bulkPoints

        self.trSize = kwargs.get("trSize")
        self.skipFirst = kwargs.get("skipFirst",54)
        self.rseed = kwargs.get("rseed",0)
        self.name = kwargs.get("name","prnn")
        self.normalize = kwargs.get("normalize",False)

        cwd = os.getcwd()
        self.filename = os.path.join(cwd, str(self.name) + '_' + str(
            self.rseed) + '_' + str(self.bulkPts*3) + '_' + str(self.trSize) + '.pth')

        # Create model

        torch.manual_seed(self.rseed)
        torch.set_default_dtype(torch.float64)

        self.model = neural_network(n_features=self.n_features, 
                                    output_length=self.n_features,
                                    bulk=self.bulkPts,
                                    dev=device).to(device)

        # Initialize from file if warmStart option specified

        if kwargs.get("loadState", False):
            print('\nLoading trained state from ', self.filename)
            self.model.load_state_dict(torch.load(self.filename))


    def calcError(self, combined, combinednndisp, nplot, nstep, ncomp=3, lb=-1.0, ub=-1.0, ):
        mse = np.array([])
        mserel = np.array([])
        for i in range(nplot):
            init = i*nstep
            end = init + nstep
            if (lb > -1.0):
                initb = init + lb
                endb = initb + (ub-lb)
            else:
                initb = init
                endb = end
            mse = np.append(mse, np.sqrt(
                ((combined[initb:endb, 3:6] - combinednndisp[initb:endb, 3:6])**2).sum(axis=1)))
            mserel = np.append(mserel, np.sqrt(((combined[initb:endb, 3:6] - combinednndisp[initb:endb, 3:6])**2).sum(
                axis=1))/np.sqrt((combined[initb:endb, 3:6]**2).sum(axis=1)))
        return mse.mean(axis=0), mserel.mean(axis=0)

    def count_parameters(self, model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            print('Name ', name, ' values\n', torch.ravel(parameter.data))
            total_params += param
        print(f"Total Trainable Params: {total_params}th")
        return total_params

    def readSmpFile(self, infilename):

        cwd = os.getcwd()
        infile = open(os.path.join(cwd, infilename), "r")
        tdata = infile.readlines()

        infile.close()
        sizetdata = len(tdata)

        for i in range(sizetdata):
            tdata[i] = tdata[i].split()

        tdata = [x for x in tdata if x != []]

        sizetdata = len(tdata)

        print("# of lines: ", sizetdata)

        y = np.array([np.array(xi) for xi in tdata])

        smpplt = y.copy()
        smpplt = np.array(smpplt, dtype=np.float64)

        return smpplt

    def readData(self, training, timesteps):
        # Getting data from sample files

        for i in range(len(training)):
            if i == 0:
                trdataset = self.readSmpFile(training[i])
            else:
                trdataset = np.append(trdataset, self.readSmpFile(training[i]))
                trdataset = trdataset.reshape(int(len(trdataset)/6), 6)

        # Get number of load cases w/ RVE

        nlctr = int(len(trdataset)/timesteps)

        print('# load cases (total): ', nlctr)
        print('# length data: ', trdataset.shape)

        # Prepare data for training the NN

        trstrain = trdataset[:, 0:3]
        trsig = trdataset[:, 3:6]

        return trstrain, trsig

    def normalize_2d(self, data):
        normdata = np.empty_like(data)
        for i in range(data.shape[1]):
            normdata[:, i] = 2.0*((data[:, i] - np.min(data[:, i])) /
                                  (np.max(data[:, i]) - np.min(data[:, i])))-1.0
        return normdata

    def writeNmlFile(self, cwd, filename, data):
        with open(os.path.join(cwd, filename), 'w') as loc:
            for i in range(data.shape[1]):
                loc.write(str(np.max(data[:, i])) +
                          ' ' + str(np.min(data[:, i])) + '\n')

    # Normalization between -1 and 1

    def applyNorm(self, cwd, filename, data):
        infile = open(os.path.join(cwd, filename), "r")
        bounds = infile.readlines()
        infile.close()
        for i in range(len(bounds)):
            bounds[i] = bounds[i].split()
        bounds = np.array(bounds, dtype=np.float64)
        normdata = np.empty_like(data)
        for i in range(data.shape[1]):
            normdata[:, i] = 2.0*((data[:, i] - bounds[i, 1]) /
                                  (bounds[i, 0] - bounds[i, 1]))-1.0
        return normdata

    def train(self, **kwargs):

        epochs = kwargs.get("epochs",100)
        earlyStop = kwargs.get("earlyStop",20)
        writeEvery = kwargs.get("writeEvery",5)
        batchSize = kwargs.get("batchSize",9)

        torch.manual_seed(self.rseed)

        self.model.train()
        trData = self.trainingData
        print('Reading training data from ', trData)
        nTimeSteps = int(trData[0][:-5][-3:])
        trstraintemp, trsigtemp = self.readData(trData, nTimeSteps)

        sequence_length = nTimeSteps
        ntr = len(trData)

        ncurves = self.trSize

        n_data = int(trsigtemp.shape[0]/(ntr*sequence_length))

        if (ncurves <= n_data):
            print("\nAttention: Using only ", ncurves, " out of ",
                  n_data, " curves to train the PRNN.\n")
            if (ncurves + self.skipFirst > n_data):
                print('\nAttention: Insufficient number of curves '
                      'available for the specified size of validation set. '
                      'Reducing it to ', n_data-ncurves, ' curves only.')
                self.skipFirst = n_data-ncurves

            rangeofidx = range(0, n_data-self.skipFirst)
            random.seed(self.rseed)
            shuffled = np.asarray(random.sample(rangeofidx, self.trSize))
            samples = shuffled + self.skipFirst
            valsamples = np.arange(0, self.skipFirst)

            # Selecting validation set

            for i in range(self.skipFirst):
                initval = valsamples[i]*sequence_length
                endval = initval + sequence_length
                if i == 0:
                    valstrain = trstraintemp[initval:endval, :]
                    valsig = trsigtemp[initval:endval, :]
                else:
                    valstrain = np.vstack(
                        [valstrain, trstraintemp[initval:endval, :]])
                    valsig = np.vstack(
                        [valsig, trsigtemp[initval:endval, :]])

            # Selecting training set

            for i in range(self.trSize):
                inittr = samples[i]*sequence_length
                endtr = inittr + sequence_length
                if i == 0:
                    trstrain = trstraintemp[inittr:endtr, :]
                    trsig = trsigtemp[inittr:endtr, :]
                else:
                    trstrain = np.vstack(
                        [trstrain, trstraintemp[inittr:endtr, :]])
                    trsig = np.vstack(
                        [trsig, trsigtemp[inittr:endtr, :]])
            n_data = ncurves
            print('Idx of samples used for validation: ', valsamples)
            print('Idx of samples used for training: ', samples, '\n')

        else:
            print("\nAttention: Number of curves available for training is lower than "
                  "expected. Using all ", n_data, " to train the PRNN. "
                  "This means that NO validation set will be considered.\n")
            trstrain = trstraintemp
            trsig = trsigtemp
            self.skipFirst = 0

        n_data = ntr*n_data

        normalizedtrstrain = self.normalize_2d(trstrain)
        cwd = os.getcwd()
        self.writeNmlFile(cwd, 'prnn.nml', trstrain)

        if self.normalize:
            trstrain = normalizedtrstrain

        x_train = trstrain.reshape([n_data, sequence_length, self.n_features])
        y_train = trsig.reshape([n_data, sequence_length, self.n_features])

        # Passing data to dataloader

        dataset = timeseries(x_train, y_train)

        # Init training loop 

        x_val = valstrain.reshape([self.skipFirst, sequence_length, self.n_features])
        y_val = valsig.reshape([self.skipFirst, sequence_length, self.n_features])

        valdataset = timeseries(x_val, y_val)
        val_loader = DataLoader(valdataset, shuffle=False)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-2)

        # Starting training loop

        stallIt = 0
        torch.autograd.set_detect_anomaly(False)
        cwd = os.getcwd()

        for i in range(epochs):

            # Shuffling the training data

            train_loader = DataLoader(dataset, shuffle=True, batch_size=batchSize)
            running_loss = 0

            for j, data in enumerate(train_loader):
                y_pred = self.model(data[:][0])
                loss = criterion(y_pred, data[:][1])
                optimizer.zero_grad()  # Clears existing gradients
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            running_loss /= (self.trSize)

            if i % writeEvery == 0:

                # Calculate loss on the validation set every N epochs

                if (self.skipFirst > 0):
                    with torch.no_grad():
                        self.model.eval()
                        running_loss_val = 0
                        for j, data in enumerate(val_loader):
                            y_predval = self.model(data[:][0])
                            loss = criterion(y_predval, data[:][1])
                            running_loss_val += loss.item()
                        running_loss_val /= (self.skipFirst)
                        self.model.train()
                else:
                    running_loss_val = running_loss

                print('Epoch ', i, ' training loss ', running_loss)
                print('Epoch ', i, ' validation loss ', running_loss_val)

                if (i == 0):
                    prev = running_loss_val

                # Only update file with best parameters if validation loss
                # is smaller than the historical best so far

                if (running_loss_val <= prev):
                    prev = running_loss_val
                    best_model_state = copy.deepcopy(
                        self.model.state_dict())
                    torch.save(best_model_state, self.filename)
                    print('Saved model in ', self.filename)
                    stallIt = 0
                else:
                    # Keeping track of how many epochs have gone
                    # without any improvement on validation set

                    stallIt += writeEvery

                if (stallIt == earlyStop):
                    loc.write('Early stopping criterion reached!\n')
                    break

        # Detail number, name and value of optimal parameters

        print('\nOptimal parameters')
        trparam = self.count_parameters(self.model)

    def eval(self, **kwargs):

        evalncurves = kwargs.get("evalncurves",5)

        trData = self.trainingData
        print('Reading testing data from ', trData)
        nTimeSteps = int(trData[0][:-5][-3:])
        trstraintemp, trsigtemp = self.readData(trData, nTimeSteps)

        sequence_length = nTimeSteps
        ntr = len(trData)
        n_data = int(trsigtemp.shape[0]/(ntr*sequence_length))

        if (evalncurves <= n_data):
            print("\nEvaluation mode: Number of curves being evaluated ", evalncurves,
                  "out of ", n_data, ". \n")
            n_data = evalncurves
            inittr = 0
            endtr = n_data*sequence_length
            inittrr = int(trsigtemp.shape[0]/ntr)
            endtrr = inittrr + endtr
            trstrain = np.vstack(
                [trstraintemp[inittr:endtr, :], trstraintemp[inittrr:endtrr, :]])
            trsig = np.vstack(
                [trsigtemp[inittr:endtr, :], trsigtemp[inittrr:endtrr, :]])
        else:
            print("\nAttention: Number of curves available is lower than "
                  "expected. Using all ", n_data, " to evaluate the PRNN.\n")
            trstrain = trstraintemp
            trsig = trsigtemp

        print('Reading parameters (weights+biases) from ', self.filename)

        n_data = ntr*n_data

        cwd = os.getcwd()
        if self.normalize:
            normalizedtrstrain = self.applyNorm(cwd, 'prnn.nml', trstrain)
            trstrain = normalizedtrstrain

        x_test = trstrain.reshape([n_data, sequence_length, self.n_features])
        y_test = trsig.reshape([n_data, sequence_length, self.n_features])

        # Pass data to dataloader

        dataset = timeseries(x_test, y_test)

        print('\nPredicting on validation/test set\n')

        test_loader = DataLoader(dataset, shuffle=False, batch_size=1)

        self.model.eval()
        for j, data in enumerate(test_loader):
            print('Progress ', j+1, '/', len(test_loader))
            test_pred = self.model(data[:][0]).cpu()
            test_pred = test_pred.detach().numpy().reshape(-1)
            if (j == 0):
                test_pred_array = np.array(test_pred)
            else:
                test_pred_array = np.append(test_pred_array, test_pred)

        combined = np.column_stack((trstrain, trsig))
        combinedprnn = np.column_stack(
            (trstrain, test_pred_array.reshape([y_test.shape[0]*y_test.shape[1], 3])))

        # Calculate error

        errabs, errrel = self.calcError(
            combined, combinedprnn, test_pred_array.shape[0], 
            sequence_length, self.n_features)

        print('\nError ', errabs)
