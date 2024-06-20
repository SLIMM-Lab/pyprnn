#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Joep Storm
August 2022

J2 material model including plasticity

Only Plane Stress is implemented
"""
import copy

import torch
import numpy as np
import os
print(f"torch version: {torch.__version__}")
class NanValueError(Exception):
    pass


class MaxIterError(Exception):
    pass


class NegativeDgamError(Exception):
    pass


torch.set_default_dtype(torch.float64)

deepcopy = True

class J2Material:
    def __init__(self,device):
        self.E = 3.13e3
        self.nu_ = 0.37
        self.fac = self.E / (1. - self.nu_)
        self.dev = device

        self.G = self.E / 2.0 / (1.0 + self.nu_)
        self.K = self.E / 3.0 / (1. - 2. * self.nu_)

        self.return_map_tol = 1e-5
        self.maxIter = 100

        self.stress_ = torch.empty((3, 1), device=self.dev)

    def configure(self, npoints):
        self.npoints = npoints
        # History changed to being handled in single tensor
        self.epsp_hist = torch.zeros((npoints, 3), device=self.dev)
        self.epspeq_hist = torch.zeros((npoints), device=self.dev) #, 1))
        self.new_epsp_hist = torch.zeros((npoints, 3), device=self.dev)
        self.new_epspeq_hist = torch.zeros((npoints), device=self.dev) #, 1))

        # 2D stiffness (Based on Jive Isotropic)
        self.el_Stiff = torch.zeros((npoints, 3, 3), device=self.dev)
        self.el_Stiff[:,0,0] = self.el_Stiff[:,1,1] = self.E / (1 - self.nu_ * self.nu_)
        self.el_Stiff[:,0,1] = self.el_Stiff[:,1,0] = (self.nu_ * self.E) / (1 - self.nu_ * self.nu_)
        self.el_Stiff[:,2,2] = 0.5 * self.E / (1 + self.nu_)

        # Initialize P matrix
        self.P_ = torch.zeros((npoints, 3, 3), device=self.dev)
        self.P_[:, 0, 0] = self.P_[:, 1, 1] = 2. / 3.
        self.P_[:, 0, 1] = self.P_[:, 1, 0] = -1. / 3.
        self.P_[:, 2, 2] = 2.

    def update(self, eps_new):
        """
        :param eps_new: [ npoints, 3  ]
        :return: stress [ npoints, 3  ]
        """
        eps_el = eps_new.view(self.npoints,-1) - self.epsp_hist
        self.eps_p_eq_0 = self.epspeq_hist

        sig_tr = torch.bmm(self.el_Stiff, eps_el.view(self.npoints,3,1))

        self.plasticity = self.isPlastic( sig_tr )

        if torch.any( self.plasticity ):
            dgam = self.findRoot()
            sig = self.compSigUpdated(dgam, sig_tr)

            # Plastic strain increment
            depsp = self.compPlastStrInc(dgam)


            # Update history
            pre = self.epsp_hist
            self.new_epsp_hist = self.epsp_hist + depsp
            self.new_epspeq_hist = self.eps_p_eq

            # Check if epsp_hist is not changed (inproper deepcopy handling)
            assert torch.allclose(pre, self.epsp_hist)
        else:
            sig = sig_tr

            self.new_epsp_hist = self.epsp_hist
            self.new_epspeq_hist  = self.epspeq_hist

        return sig[:, :, 0]


    def getHistory (self):
        return self.new_epspeq_hist
    
    def commit ( self ):
        self.epsp_hist = self.new_epsp_hist
        self.epspeq_hist = self.new_epspeq_hist

    
    def sigma_C(self, x):  # TODO include look-up table external file
        return 64.80 - 33.60 * torch.exp(x / -0.003407)


    def sigma_C_deriv(self, x):
        return 9862.048723216907*torch.exp(x/-0.003407)

    def evalTrial_(self):
        # fast version of eval, without computing all dgam dependent variables
        sigY = self.sigma_C(self.eps_p_eq_0)

        self.yieldVal = 0.5 * self.xi_tr - sigY ** 2 / 3.0
        return self.yieldVal

    def evalXi(self, dgam):
        f1 = 1. + self.fac * dgam / 3.
        f2 = 1. + 2. * self.G * dgam
        xi = self.A11_tr / (6 * f1 ** 2) + (0.5 * self.A22_tr + 2. * self.A33_tr) / f2 ** 2
        xi_der = - self.A11_tr * self.fac / (9 * f1 ** 3) - 2 * self.G * (self.A22_tr + 4 * self.A33_tr) / f2 ** 3
        return xi, xi_der

    def evalYield_(self, dgam):
        self.xi, self.xi_der = self.evalXi(dgam)
        self.eps_p_eq = self.eps_p_eq_0 + dgam * torch.sqrt(2. * self.xi / 3.)
        sigY = self.sigma_C(self.eps_p_eq)
        return 0.5 * self.xi - sigY ** 2 / 3.

    def evalYieldDer(self, dgam):
        sigY = self.sigma_C(self.eps_p_eq)
        H = self.sigma_C_deriv(self.eps_p_eq)
        xi_sqrt = torch.sqrt(self.xi)
        H_bar = 2 * sigY * H * 0.81649658092 * (xi_sqrt + dgam * self.xi_der / (2 * xi_sqrt))
        return self.xi_der / 2 - H_bar / 3

    def isPlastic(self, sig_tr):

        self.A11_tr = (sig_tr[:,0,0] + sig_tr[:,1,0]) ** 2
        self.A22_tr = (sig_tr[:,1,0] - sig_tr[:,0,0]) ** 2
        self.A33_tr = sig_tr[:,2,0] ** 2
        self.xi_tr = self.A11_tr / 6.0 + 0.5 * self.A22_tr + 2.0 * self.A33_tr
        return self.evalTrial_() >= self.return_map_tol

    def findRoot(self):
        # dgam needs to be torch.float64, and everything that depends on it (xi) will follow to be 64.
        # Otherwise problems with convergence.
        dgam = torch.zeros(self.npoints, device=self.dev, dtype=torch.float64)

        oldddgam = torch.ones_like(dgam) * -1

        oldyieldval = self.yieldVal

        # TODO note: now iterate all plastic ones for the same number of iterations.
        # Sould instead stop updating those that have already converged?
        # (expect the computational cost to be a negligible difference, I pass full tensors anyway)
        for i in range(self.maxIter):
            # print(f"iter: {i}")

            dgam_clone = dgam.clone()   # Either clone here to prevent in-place operation, or in both evalYield & evalYieldDer clone individually
            yieldval = self.evalYield_(dgam_clone)

            # If not plastic, multiply by False (=0)
            # If plastic, multiply by the True  (=1)
            yieldval = yieldval * self.plasticity

            converged_points = torch.abs(yieldval) < self.return_map_tol

            if torch.all( converged_points ):  # All converged
                break
            elif i == self.maxIter - 1:
                raise MaxIterError

            yield_deriv = self.evalYieldDer(dgam_clone)

            ddgam = yieldval / yield_deriv
            # ddgam is identical for first RM step as in non-vectorized form

            if torch.any(ddgam.isnan()):
                raise NanValueError

            # Modified 23-05-2023
            # Also, tolerance divergene_tol was -1e14, changed to 0
            if torch.any(ddgam * oldddgam < 0 ): 
                # In batch mode, find if any fulfill both divergence criteria
                div_criteria_yield = oldyieldval * yieldval < 0
                div_criteria_ddgam = torch.abs(ddgam) > torch.abs(oldddgam)
                div_criteria = div_criteria_yield * div_criteria_ddgam
                if torch.any(div_criteria):
                    print(" --------- divergence detection! --------- ")
                    for i, criteria in enumerate(div_criteria):
                        if criteria:
                            #  there might be an inflection point around the root
                            #  use linear interpolation rather than linearization
                            ddgam[i] = -oldddgam[i] * yieldval[i] / (yieldval[i] - oldyieldval[i])

            # Multiply ddgam with opposite of converged_points to only adapt those of dgam which are not yet converged.
            # Without this, dgams become negative faster if they keep iterating while converged
            # (Note, some still become negative; I lowered tolerance below form -1e-12 to -1e-10 to see if sufficient)
            dgam -= ddgam * ~converged_points

            # Store values for next iter
            oldddgam = ddgam
            oldyieldval = yieldval

        if torch.any(dgam[torch.nonzero(dgam)] < -1e-10):
            raise NegativeDgamError

        dgam = dgam.to(torch.float)
        return dgam

    def compSigUpdated(self, dgam, sig_tr):
        A = self.getAMatrix(dgam)
        self.stress_ = torch.bmm(A, sig_tr)
        return self.stress_

    def compPlastStrInc(self, dgam):
        return dgam.view(-1, 1) * torch.bmm(self.P_, self.stress_)[:,:,0]

    def getAMatrix(self, dgam):

        A_mat = torch.zeros((self.npoints, 3, 3), device=self.dev)

        A_mat[:, 0, 0] = A_mat[:, 1, 1] = ((3. * (1. - self.nu_) / (3. * (1. - self.nu_) + self.E * dgam)) + (1. / (1 + 2. * self.G * dgam))) / 2
        A_mat[:, 1, 0] = A_mat[:, 0, 1] = ((3. * (1. - self.nu_) / (3. * (1. - self.nu_) + self.E * dgam)) - (1. / (1 + 2. * self.G * dgam))) / 2

        A_mat[:,2, 2] = 1. / (1 + 2. * self.G * dgam)

        return A_mat
