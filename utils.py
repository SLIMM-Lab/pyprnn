#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes for training RNNs and PRNNs"""


import os
import numpy as np

import torch
import copy
import pandas


class Normalizer:
    """Normalization for strain features.

    Scales strain data to a [-1,1] interval
    """

    def __init__(self,X):
        self.min = X.min(axis=0).values
        self.max = X.max(axis=0).values

    def normalize(self,x):
        return 2.0 * ((x - self.min) / (self.max-self.min)) - 1.0


class StressStrainDataset(torch.utils.data.Dataset):
    """Custom dataset for handling stress-strain paths.

    Dataset is loaded with pandas and stress-strain pairs are
    split into paths of 'seq_length' time steps. Normalization
    is optional and the normalizer can be inherited from another
    dataset (e.g. to handle an entirely new test set).
    """

    def __init__(self, filename, features, targets, seq_length, **kwargs):
        df = pandas.read_csv(filename, delim_whitespace=True,header=None)

        self.seq_length = seq_length

        self.X = torch.tensor(df[features].values, dtype=torch.float64)
        self.T = torch.tensor(df[targets].values,  dtype=torch.float64)

        self.normalize_features = kwargs.get('normalize_features',False)

        if self.normalize_features:
            self._normalizer = kwargs.get('normalizer',Normalizer(self.X))

    def __len__(self):
        return int(self.X.shape[0]/self.seq_length)

    def __getitem__(self,idx):
        start = idx * self.seq_length
        end   = start + self.seq_length

        x = self.X[start:end,:]
        t = self.T[start:end,:]

        if self.normalize_features:
            return self._normalizer.normalize(x), t
        else:
            return x, t

    def get_normalizer(self):
        if self.normalize_features:
            return self._normalizer
        else:
            return None


class Trainer ():
    """Class for handling standard network training tasks.

    Wraps an existing model inherited from 'torch.nn.Module' and
    performs training and evaluation tasks. Loss function and optimizer 
    are fixed but training and validation can be performed with different
    DataLoaders by calling the 'train()' function multiple times.

    Early stopping is implemented with adjustable patience. At the end
    of training, the network with lowest historical validation error
    is stored.

    Evaluation can be done on a testset DataLoader. The (time) average
    error for each sample (path) is reported, as well as the average
    over the full test set.

    Network and optimizer states can be saved and loaded from files.
    """

    def __init__(self, model, **kwargs):
        self._model = model

        self._epoch = 0
        self._criterion = kwargs.get('loss',torch.nn.MSELoss())
        self._optimizer = kwargs.get(
                'optimizer',
                torch.optim.Adam(self._model.parameters()))

        total_params = 0
        for name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()

        print('Total parameter count:', total_params,'\n')

    def train(self, training_loader, validation_loader, **kwargs):
        self._model.train()

        epochs   = kwargs.get('epochs',100)
        patience = kwargs.get('patience',20)
        interval = kwargs.get('interval',1)
        verbose  = kwargs.get('verbose',True)

        stall_iters = 0
        torch.autograd.set_detect_anomaly(False)

        for i in range(epochs):
            running_loss = 0

            for x, t in training_loader:
                y = self._model(x)
                loss = self._criterion(y, t)
                self._optimizer.zero_grad() 
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item()

            running_loss /= len(training_loader)

            self._epoch += 1

            if i < interval or i % interval == 0:
                with torch.no_grad():
                    self._model.eval()
                    running_loss_val = 0
                    for x, t in validation_loader:
                        y = self._model(x)
                        loss = self._criterion(y, t)
                        running_loss_val += loss.item()
                    running_loss_val /= len(validation_loader)
                    self._model.train()

                if verbose:
                    print('Epoch',
                          self._epoch,
                          'training loss',
                          running_loss)

                if verbose:
                    print('Epoch',
                          self._epoch,
                          'validation loss',
                          running_loss_val)

                if (self._epoch == 1):
                    self._best_val = running_loss_val

                if (running_loss_val <= self._best_val):
                    self._best_val = running_loss_val
                    self._best_state_dict = copy.deepcopy(self._model.state_dict())
                    stall_iters = 0

                    if verbose:
                        print('The best historical model has been updated.',
                              'Resetting early stop counter.')
                else:
                    if i <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if (stall_iters >= patience):
                    if verbose:
                        print('Early stopping criterion reached.')
                    break

        print('End of training.')

    def eval(self, test_loader, **kwargs):
        criterion  = kwargs.get('loss',self._criterion)
        verbose    = kwargs.get('verbose',True)

        live_state = copy.deepcopy(self._model.state_dict())

        self._model.load_state_dict(self._best_state_dict)

        combined_loss = 0.0

        self._model.eval()
        for j, (x,t) in enumerate(test_loader):            
            
            y = self._model(x)
            loss = criterion (y, t)
            combined_loss += loss.item()

            if verbose:
                print('Loss for test batch ',
                      j+1, '/', len(test_loader),':',loss.item())
            
        if verbose:
            print('Aggregated test set loss:', 
                  combined_loss/len(test_loader))

        self._model.load_state_dict(live_state)
        
        return combined_loss/len(test_loader)

    def save(self, filename):
        torch.save({
                    'epoch': self._epoch,
                    'best_val': self._best_val,
                    'model_state_dict': self._model.state_dict(),
                    'best_state_dict': self._best_state_dict,
                    'optimizer_state_dict': self._optimizer.state_dict()
                   }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)

        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._best_state_dict = checkpoint['best_state_dict']
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
