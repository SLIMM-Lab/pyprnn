#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shallow GRU and LSTM networks

Single-layer encoder-decoder with a latent RNN layer. 
Gaussian dropout is used for regularization, with intensity
automatically tweaked during training by maximizing an
appropriate Evidence Lower Bound (ELBO). These networks can
be used to compare PRNNs against what is currently the most
popular approach for path-dependent surrogate constitutive modeling. 
"""

import torch


class LSTM(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_latents, dropout=False):
        super(LSTM,self).__init__()

        self.n_features = n_features
        self.n_latents  = n_latents
        self.n_outputs  = n_outputs
        self.n_layers   = 1

        print('------ LSTM model summary ------')
        print('Input size', self.n_features)
        print('Latent state size', self.n_latents)
        print('LSTM cell state size', self.n_latents)
        print('Output size', self.n_outputs)
        print('--------------------------------')

        self.lstm = torch.nn.LSTM(input_size=self.n_features,
                                  hidden_size=self.n_latents,
                                  batch_first=True,
                                  num_layers=self.n_layers)

        self.linear = torch.nn.Linear(in_features=self.n_latents,
                                      out_features=self.n_outputs)

        if dropout:
            self.dropout = GaussianDropout()
        else:
            self.dropout = None

    def forward(self,x):
        batch_size = x.shape[0]

        h0 = torch.zeros(self.n_layers,
                         batch_size,
                         self.n_latents,
                         requires_grad=True)
        c0 = torch.zeros(self.n_layers,
                         batch_size,
                         self.n_latents,
                         requires_grad=True)

        output, (hn, _) = self.lstm(x, (h0,c0))

        if self.dropout:
            output = self.dropout(output)

        output = self.linear(output)

        return output


class GRU(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_latents, dropout=False):
        super(GRU,self).__init__()

        self.n_features = n_features
        self.n_latents  = n_latents
        self.n_outputs  = n_outputs
        self.n_layers   = 1

        print('------ GRU model summary ------')
        print('Input size', self.n_features)
        print('Latent state size', self.n_latents)
        print('Output size', self.n_outputs)
        print('-------------------------------')

        self.gru = torch.nn.GRU(input_size=self.n_features,
                                  hidden_size=self.n_latents,
                                  batch_first=True,
                                  num_layers=self.n_layers)

        self.linear = torch.nn.Linear(in_features=self.n_latents,
                                      out_features=self.n_outputs)

        if dropout:
            self.dropout = GaussianDropout()
        else:
            self.dropout = None

    def forward(self,x):
        batch_size = x.shape[0]

        h0 = torch.zeros(self.n_layers,
                         batch_size,
                         self.n_latents,
                         requires_grad=True)

        output, hn = self.gru(x, h0)

        if self.dropout:
            output = self.dropout(output)

        output = self.linear(output)

        return output

class GaussianDropout(torch.nn.Module):
    def __init__(self):
        super(GaussianDropout, self).__init__()

        self.rate = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self._sp  = torch.nn.Softplus()

    def forward(self, x):
        if self.training:
            stddev = self._sp(self.rate) 

            noise = torch.randn_like(x) * stddev + 1

            return x * noise
        else:
            return x

class ELBOLoss(torch.nn.Module):
    def __init__(self, model):
        super(ELBOLoss, self).__init__()

        self._model = model
        self._mse   = torch.nn.MSELoss()
        self._sp    = torch.nn.Softplus()

    def forward(self, pred, target):
        mse_loss = self._mse(pred, target)
        kl_loss = 0.5 * torch.log(1+torch.pow(self._sp(self._model.rate),-2))

        if self.training:
            return mse_loss + kl_loss.squeeze()
        else:
            return mse_loss
