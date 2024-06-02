"""
@author: Joep Storm
August 2022

J2 material model including plasticity

Only Plane Stress is implemented
"""
import copy

import torch
print(f"torch version: {torch.__version__}")
import math  #Some exp function?
# import torchviz
# from torchviz import make_dot

torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)

deepcopy = True

class J2Material:
    def __init__(self):
        # Material properties
        
        self.E = 3.13e3
        self.nu_ = 0.37

        # self.rmTolerance_ = 1.e-10 # 1.e-10
        self.rmMaxIter_ = 25

        # self.nu_pd = 0
        self.eps_cur = torch.zeros((6))

        self.epsp_hist = []
        self.new_epsp_hist = []
        self.latest_hist = []

        self.y_ = []

        # Compute elastic stiffness
        self.el_Stiff = torch.zeros((6, 6), dtype=torch.float64)

        d = (1.00 + self.nu_) * (1.00 - 2.00 * self.nu_)
        self.el_Stiff[0,0] = self.el_Stiff[1,1] = self.el_Stiff[2,2] = self.E * (1.0 - self.nu_) / d
        self.el_Stiff[0, 1] = self.el_Stiff[1, 0] = self.el_Stiff[0, 2] = self.el_Stiff[2, 0] = self.el_Stiff[1, 2] = self.el_Stiff[2, 1] = (self.E * self.nu_ / d)
        self.el_Stiff[3, 3] = self.el_Stiff[4, 4] = self.el_Stiff[5, 5] = 0.5 * self.E / (1.0 + self.nu_)
        # TODO: This next line can likely be removed. Don't need to compute w.r.t. el_stiff
        # self.el_Stiff = self.el_Stiff.clone().detach().requires_grad_(True)

    def getHistory(self, ip):
        return self.latest_hist[ip].HistVec()

    def setHistory(self, eps_plastic, eps_p_eq, ip):
        if not deepcopy:
            self.epsp_hist[ip].eps_plastic = eps_plastic
            self.epsp_hist[ip].eps_p_eq = eps_p_eq
            self.latest_hist[ip] = self.epsp_hist[ip]
            self.new_epsp_hist[ip] = self.epsp_hist[ip]
        else:
            self.epsp_hist[ip].eps_plastic = eps_plastic.clone()
            self.epsp_hist[ip].eps_p_eq = eps_p_eq.clone()
            self.latest_hist[ip].eps_plastic = eps_plastic.clone()
            self.latest_hist[ip].eps_p_eq = eps_p_eq.clone()
            self.new_epsp_hist[ip].eps_plastic = eps_plastic.clone()
            self.new_epsp_hist[ip].eps_p_eq = eps_p_eq.clone()


    # def resetDiffGraph(self, ipoint):
    #     if not deepcopy:
    #         self.epsp_hist[ipoint].eps_plastic = self.latest_hist[ipoint].eps_plastic.clone().detach().requires_grad_(True)
    #         self.epsp_hist[ipoint].eps_p_eq    = self.latest_hist[ipoint].eps_p_eq.clone().detach().requires_grad_(True)
    #     else:
    #         self.epsp_hist[ipoint].eps_plastic = self.latest_hist[ipoint].eps_plastic.clone() #.detach().requires_grad_(True)
    #         self.epsp_hist[ipoint].eps_p_eq    = self.latest_hist[ipoint].eps_p_eq.clone() #.detach().requires_grad_(True)

    def configure(self, npoints):
        for i in range(npoints):
            self.epsp_hist.append( self.Hist() )
            self.new_epsp_hist.append( self.Hist() )
            self.latest_hist.append( self.Hist() )
            self.y_.append( self.YieldFunc(self.E, self.nu_))

    def reduce3DVector_(self, v, t):
        # reduce a full 3D tuple to a 2D or 3D vector
        if (v.shape[0] == 1):
            v[0] = t[0]
        elif (v.shape[0] == 3):  # Plane strain excl. s_zz
            v[0] = t[0]
            v[1] = t[1]
            v[2] = t[3]


    def update(self, eps_new, ip_point):  # Do I make it possible to give individual ip_points?
        eps_el = torch.zeros((6), dtype=torch.float64)

        eps_el[0] = eps_new[0] - self.epsp_hist[ip_point].eps_plastic[0]
        eps_el[1] = eps_new[1] - self.epsp_hist[ip_point].eps_plastic[1]
        eps_el[2] = -self.nu_ / (1 - self.nu_) * (eps_el[0] + eps_el[1])
        eps_el[3] = eps_new[2] - self.epsp_hist[ip_point].eps_plastic[3]

        eps_p_eq_0 = self.epsp_hist[ip_point].eps_p_eq

        sig_tr = torch.matmul( self.el_Stiff, eps_el )
        
        plasticity = self.y_[ip_point].isPlastic( sig_tr, eps_p_eq_0 )
        if plasticity:
            dgam = self.y_[ip_point].findRoot(torch.tensor([0.0]))

            sig = self.y_[ip_point].compSigUpdated(dgam, sig_tr)        # Stress

            # Plastic strain increment
            depsp = self.y_[ip_point].compPlastStrInc(dgam)

            # Update history
            pre = self.epsp_hist[ip_point].eps_plastic
            self.new_epsp_hist[ip_point].eps_plastic = self.epsp_hist[ip_point].eps_plastic + depsp.view(-1)

            # Check if epsp_hist is not changed (inproper deepcopy handling)
            assert torch.allclose(pre, self.epsp_hist[ip_point].eps_plastic)

            self.new_epsp_hist[ip_point].eps_p_eq = self.y_[ip_point].getEpspeq()
        else:
            sig = sig_tr
            if deepcopy:
                self.new_epsp_hist[ip_point].eps_plastic = self.epsp_hist[ip_point].eps_plastic.clone()
                self.new_epsp_hist[ip_point].eps_p_eq = self.epsp_hist[ip_point].eps_p_eq.clone()
            else:
                self.new_epsp_hist[ip_point].eps_plastic = self.epsp_hist[ip_point].eps_plastic
                self.new_epsp_hist[ip_point].eps_p_eq = self.epsp_hist[ip_point].eps_p_eq

        stress = torch.zeros_like(eps_new)

        self.reduce3DVector_(stress, sig)

        if deepcopy:
            self.latest_hist[ip_point].eps_plastic = self.epsp_hist[ip_point].eps_plastic.clone()
            self.latest_hist[ip_point].eps_p_eq = self.epsp_hist[ip_point].eps_p_eq.clone()
        else:
            self.latest_hist[ip_point] = self.new_epsp_hist[ip_point]

        return stress

    def commit ( self, ipoint ):
        if not deepcopy:
            self.epsp_hist[ipoint] = self.new_epsp_hist[ipoint]
            self.latest_hist[ipoint] = self.epsp_hist[ipoint]
        else:
            self.epsp_hist[ipoint].eps_plastic = self.new_epsp_hist[ipoint].eps_plastic.clone()
            self.epsp_hist[ipoint].eps_p_eq = self.new_epsp_hist[ipoint].eps_p_eq.clone()
            self.latest_hist[ipoint].eps_plastic = self.epsp_hist[ipoint].eps_plastic.clone()
            self.latest_hist[ipoint].eps_p_eq = self.epsp_hist[ipoint].eps_p_eq.clone()

    class Hist:
        def __init__(self):
            self.eps_plastic = torch.zeros(6) #, requires_grad=True)
            self.eps_p_eq = torch.zeros(1, requires_grad=True)

        def HistVec(self):
            return self.eps_plastic, self.eps_p_eq


    class YieldFunc:
        def __init__(self, young, poisson):
            self.young = young
            self.poisson = poisson

            self.G = self.young / 2.0 / (1.0 + self.poisson)
            self.K = self.young / 3.0 / (1. - 2. * self.poisson)

            self.eps_p_eq = 0.0
            self.eps_p_eq_0 = 0.0
            self.deps_p_eq = 0.0

            self.return_map_tol = 1e-7
            self.maxIter = 100

            self.P_ = torch.zeros((3, 3))

            self.P_[0, 0] = self.P_[1, 1] = 2. / 3.
            self.P_[0, 1] = self.P_[1, 0] = -1. / 3.
            self.P_[2, 2] = 2.

            self.stress_ = torch.empty((3, 1)) #, requires_grad=True)

        def sigma_C(self, x):  # TODO include look-up table external file
             # Material properties (yield function)
            return 64.80 - 33.60 * torch.exp(x / -0.003407)


        def sigma_C_deriv(self, x):
            return 9862.048723216907*torch.exp(x/-0.003407)

        def findBounds(self):
            gmin = 0.0
            gmax = -1.0
            fmax = -1.0

            fmin = self.evalTrial_()

            if (fmin <= 0.0):
                raise Exception('negative fmin')

            dgam = 1.e-16

            while (gmax < 0.0):
                crit = self.evalYield_(dgam)

                if (crit > 0.0):
                    gmin = dgam
                    fmin = crit
                else:
                    gmax = dgam
                    fmax = crit

                dgam = dgam * 10

            return gmin, gmax, fmin, fmax

        def evalTrial_(self):
            # fast version of eval, without computing all dgam dependent variables
            sigY = self.sigma_C(self.eps_p_eq_0)

            return 0.5 * self.xi_tr - sigY**2 / 3.0

        def evalXi(self, dgam):
            fac = self.young / (1. - self.poisson)
            f1 = 1. + fac * dgam / 3.
            f2 = 1. + 2. * self.G * dgam
            xi = self.A11_tr / (6*f1**2) + (0.5 * self.A22_tr + 2. * self.A33_tr) / f2**2
            xi_der = - self.A11_tr * fac / (9*f1**3) - 2 * self.G * (self.A22_tr + 4*self.A33_tr) / f2**3
            return xi, xi_der

        def evalYield_(self, dgam):
            xi, xid_er  = self.evalXi(dgam)
            self.eps_p_eq = self.eps_p_eq_0 + dgam.clone() * torch.sqrt(2. * xi / 3.)
            sigY = self.sigma_C(self.eps_p_eq)
            return 0.5 * xi - sigY**2 / 3.

        def evalYieldDer(self, dgam):
            xi, xi_der = self.evalXi( dgam )
            sigY = self.sigma_C(self.eps_p_eq)
            H = self.sigma_C_deriv( self.eps_p_eq )
            H_bar = 2*sigY * H * torch.sqrt(torch.tensor([2/3])) * ( torch.sqrt(xi) + dgam.clone() * xi_der / (2*torch.sqrt(xi))  )
            # Clone required for backprop
            return xi_der / 2 - H_bar / 3

        def isPlastic(self, sig_tr, eps_p_eq_0):
            self.eps_p_eq_0 = eps_p_eq_0
            self.A11_tr = ( sig_tr[0] + sig_tr[1] ) ** 2
            self.A22_tr = ( sig_tr[1] - sig_tr[0] ) ** 2
            self.A33_tr = sig_tr[3] ** 2

            self.xi_tr = self.A11_tr/6.0 + 0.5*self.A22_tr + 2.0*self.A33_tr

            return self.evalTrial_() >= self.return_map_tol

        def getEpspeq(self):
            return self.eps_p_eq

        def findRoot(self, dgam0):
            dgam = dgam0
            # dgam = Variable(dgam, requires_grad=True)
            oldddgam = torch.tensor([-1])
            oldyieldval = self.evalTrial_()

            for i in range(self.maxIter):
                yieldval = self.evalYield_(dgam)

                if abs(yieldval) < self.return_map_tol:  # Converged
                    break
                elif i == self.maxIter - 1:
                    raise Exception("No convergence")

                yield_deriv = self.evalYieldDer(dgam)

                ddgam = yieldval / yield_deriv

                if torch.isnan(ddgam):
                    raise Exception("nan")

                if ddgam * oldddgam < 0:    # Divergence detection
                    if oldyieldval * yieldval < 0 and abs(ddgam) > abs(oldddgam):
                        print(" --------- divergence detection! --------- ")
                        ddgam = -oldddgam * yieldval / (yieldval - oldyieldval)
                        print(f"ddgam: {ddgam.item():.12e} ")
                dgam -= ddgam
                oldddgam = ddgam
                oldyieldval = yieldval

            if dgam < 1e-12:
                raise Exception("negative dgam")
            return dgam

        def compSigUpdated(self, dgam, sig_tr):
            sig_tr3 = self.t6_to_t3_(sig_tr)
            A = self.getAMatrix(dgam)
            self.stress_ = torch.matmul(A, sig_tr3)
            sig = self.t3_to_t6_(self.stress_)
            return sig

        def compPlastStrInc(self, dgam):
            depsp3 = dgam * torch.matmul(self.P_, self.stress_)
            depsp = self.ep3_to_ep6_(depsp3)
            return depsp

        def getAMatrix(self, dgam):
            A_mat = torch.zeros((3,3))
            A_mat[0, 0] = ((3. * ( 1. - self.poisson ) / ( 3. * ( 1. - self.poisson ) + self.young * dgam )) + (1. / (1 + 2. * self.G * dgam) ) ) / 2
            A_mat[1, 0] = ((3. * ( 1. - self.poisson ) / ( 3. * ( 1. - self.poisson ) + self.young * dgam )) - (1. / (1 + 2. * self.G * dgam) ) ) / 2
            A_mat[0, 1] = A_mat[1, 0]
            A_mat[1, 1] = ((3. * ( 1. - self.poisson ) / ( 3. * ( 1. - self.poisson ) + self.young * dgam )) + (1. / (1 + 2. * self.G * dgam) ) ) / 2
            A_mat[2, 2] = 1. / (1 + 2. * self.G * dgam)
            return A_mat

        def t6_to_t3_(self, t6):
            t3 = torch.empty((3, 1)) #, requires_grad=True)
            t3[0] = t6[0]
            t3[1] = t6[1]
            t3[2] = t6[3]
            return t3

        def t3_to_t6_(self, t3):
            t6 = torch.zeros((6, 1)) #, requires_grad=True)
            t6[0] = t3[0]
            t6[1] = t3[1]
            t6[3] = t3[2]
            return t6

        def ep3_to_ep6_(self, ep3):
            ep6 = torch.zeros((6, 1))
            ep6[0] = ep3[0]
            ep6[1] = ep3[1]
            ep6[2] = -ep3[0] - ep3[1]  # out-of-plane component
            ep6[3] = ep3[2]

            return ep6

        # TODO pytorcherize
        def findRootimp(self, gmin, gmax, fmin, fmax):
            try:
                dgam0 = self.estimateRoot_(gmin, gmax, fmin, fmax)
                dgam = self.findRoot(dgam0)
            except:
                gmin, gmax, fmin, fmax = self.improveBounds_(gmin, gmax, fmin, fmax)
                dgam = self.findRootimp2(gmin, gmax, fmin, fmax)
            return dgam

        def findRootimp2(self, gmin, gmax, fmin, fmax):
            try:
                dgam0 = self.estimateRoot_(gmin, gmax, fmin, fmax)
                dgam = self.findRoot(dgam0)
            except:
                gmin, gmax, fmin, fmax = self.improveBounds_(gmin, gmax, fmin, fmax)
                dgam = self.findRootimp(gmin, gmax, fmin, fmax)
            return dgam

        def estimateRoot_(self, gmin, gmax, fmin, fmax):
            return gmin + fmin * (gmax - gmin) / (fmin - fmax)

        def improveBounds_(self, gmin, gmax, fmin, fmax):
            npval = 10
            dg = float((gmax - gmin) / npval)
            dgam = gmin

            while (dgam < gmax):
                dgam = dgam + dg
                crit = self.evalYield_(dgam)

                if (crit > 0.0):
                    gmin = dgam
                    fmin = crit
                else:
                    gmax = dgam
                    fmax = crit
                    break

            return gmin, gmax, fmin, fmax

