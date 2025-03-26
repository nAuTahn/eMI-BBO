import numpy as np
from numpy import sqrt, exp, log
from bisect import bisect_left
from scipy.stats import chi2, norm

#import cocoex, cocopp  # experimentation and post-processing modules
#import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
#import cma
#import json
#import math


class lbvdcma:
    def __init__(self, func, xmean0, sigma0, domain_int, margin, lamb, **kwargs):
        if 'seed' in kwargs.keys():
            np.random.seed(kwargs['seed'])

        self.N = len(xmean0)
        self.lam = lamb
        wtemp = np.array([
            np.log((self.lam + 1.) / 2.0) - np.log(1 + i)
            for i in range(self.lam // 2)
        ])
        self.w = kwargs.get('w', wtemp / np.sum(wtemp))
        self.sqrtw = np.sqrt(self.w)
        self.mueff = 1.0 / (self.w**2).sum()
        self.mu = self.w.shape[0]

        self.cm = 1.0
        self.cfactor = kwargs.get('cfactor', max((self.N - 5) / 6.0, 0.5))
        self.cc = kwargs.get('cc', (4 + self.mueff / self.N) /
                             (self.N + 4 + 2 * self.mueff / self.N))
        self.cone = kwargs.get('cone', self.cfactor * 2 / (
            (self.N + 1.3)**2 + self.mueff))
        self.cmu = kwargs.get('cmu',
                              min(1 - self.cone, self.cfactor * 2 *
                                  (self.mueff - 2 + 1 / self.mueff) / (
                                      (self.N + 2)**2 + self.mueff)))
        self.ssa = kwargs.get('ssa', 'MCSA')  # or 'TPA'

        self.flg_injection = False

        if self.ssa == 'TPA':
            self.cs = kwargs.get('cs', 0.3)
            self.ds = kwargs.get('ds', np.sqrt(self.N))
            self.dx = np.zeros(self.N)
            self.ps = 0
        elif self.ssa == 'MCSA':
            self.cs = kwargs.get('cs',
                                 0.5 / (np.sqrt(self.N / self.mueff) + 1.0))
            self.ds = kwargs.get(
                'ds',
                1 + 2.0 * max(0, np.sqrt(
                    (self.mueff - 1) / (self.N + 1)) - 1) + self.cs)
            self.ps = np.zeros(self.N)
        else:
            raise NotImplementedError('ssa = ' + self.ssa +
                                      ' is not implemented.')

        self.func = func
        self.xmean = xmean0
        self.sigma = sigma0
        self.dim_co = self.N // 2
        self.dim_int = self.N // 2
        self.domain_int = domain_int
        self.norm_ci = norm.ppf(1. - margin)
        self.dig = np.ones(self.N)

        self.iters = 0

        for i in range(self.dim_int):
            self.domain_int[i].sort()
            assert (len(domain_int[i]) >= 2), f"The number of elements of the domain in an integer variable must be greater than or equal to 2"
        self.lim = [[(domain_int[i][j] + domain_int[i][j + 1]) / 2. for j in range(len(domain_int[i]) - 1)] for i in range(self.dim_int)]

        self.dvec = np.ones(self.N)
        self.vvec = np.random.normal(0., 1., self.N) / float(np.sqrt(self.N))
        self.norm_v2 = np.dot(self.vvec, self.vvec)
        self.norm_v = np.sqrt(self.norm_v2)
        self.vn = self.vvec / self.norm_v
        self.vnn = self.vn**2

        self.pc = np.zeros(self.N)
        self.chiN = np.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 /
                                       (21.0 * self.N * self.N))
        self.sqrtcmuw = np.sqrt(self.cmu * self.w)
        self.fbest = self.func(self.xmean)
        self.x_best = self.xmean #add
        self.neval = 0
        self.is_success = False

        self.ftarget = kwargs.get('ftarget', 1e-10)
        self.maxeval = kwargs.get('maxeval', 10 * 10 * 10 * 10 * self.N)
        
        self.batch_evaluation = kwargs.get('batch_evaluation', False)

    def run(self):

        itr = 0
        satisfied = False
        while not satisfied:
            itr += 1
            self._onestep()
            satisfied, condition = self._check()
        return self.is_success, self.x_best, self.fbest, self.neval

    def _mahalanobis_square_norm(self, dx):
        """Square norm of dx w.r.t. C
        Parameters
        ----------
        dx : numpy.ndarray (1D)
        Returns
        -------
        square of the Mahalanobis distance dx^t * D^{-1} * (I + v*v^t)^{-1} * D^{-1} * dx,
            where (I + v*v^t)^{-1} = I - (1 + ||v||^2)^{-1} * v * v^t
        """
        ddx = dx / self.dvec
        return (ddx * ddx).sum() - np.dot(ddx, self.vvec)**2 / (1.0 +
                                                                self.norm_v2)

    def _onestep(self):
        self.iters += 1

        # Sampling
        arz = np.random.randn(self.lam, self.N)
        ary = self.dvec * (arz + (np.sqrt(1.0 + self.norm_v2) - 1.0) *
                           np.outer(np.dot(arz, self.vn), self.vn))
        if self.flg_injection:
            mnorm = self._mahalanobis_square_norm(self.dx)
            dy = (np.linalg.norm(np.random.randn(self.N)) /
                  np.sqrt(mnorm)) * self.dx
            ary[0] = dy
            ary[1] = -dy
        arx = self.xmean + self.sigma * ary

        l_close = np.array([
            self.lim[i][min(len(self.lim[i]) - 1, max(0, bisect_left(self.domain_int[i], self.xmean[self.dim_co + i]) - 1))]
            for i in range(self.dim_int)])#.reshape(self.dim_int,1)

        xbar = np.array([[
            arx[i][j] if j < self.dim_co else
            self.domain_int[j - self.dim_co][bisect_left(self.lim[j - self.dim_co], arx[i][j])]
            for j in range(self.N)] for i in range(self.lam)]) # real candidates

        # Evaluation
        if self.batch_evaluation:
            arf = self.func(xbar)
        else:
            arf = np.zeros(self.lam)
            for i in range(self.lam):
                arf[i] = self.func(xbar[i])            
        self.neval += self.lam
        self.arf = arf
        idx = np.argsort(arf)
        f_best = arf[idx[0]]
        x_best = xbar[idx[0]]
        sary = ary[idx[:self.mu]]

        if f_best < self.fbest:
            self.fbest = f_best
            self.x_best = x_best

        # Update xmean
        self.dx = np.dot(self.w,
                         arx[idx[:self.mu]]) - np.sum(self.w) * self.xmean
        

        # Bias
        eta_m = np.ones(self.N)
        ci = (self.norm_ci * self.sigma * np.sqrt(self.dig))[self.dim_co:]
        ci_up = self.xmean[self.dim_co:] + ci
        ci_low = self.xmean[self.dim_co:] - ci
        resolution = np.array([bisect_left(self.lim[i], ci_up[i]) - bisect_left(self.lim[i], ci_low[i]) for i in range(self.dim_int)])
        condition_bias = (resolution <= 1) & ~(((self.dx)[self.dim_co:] < 0.0) ^ (self.xmean[self.dim_co:] - l_close < 0.0))
        eta_m[self.dim_co:][np.where(condition_bias)] += 1.0
        self.cm = eta_m
        self.xmean += eta_m * self.dx
        #self.xmean += self.dx

        # Update sigma
        if self.ssa == 'MCSA':
            ymean = np.dot(self.w, sary) / self.dvec
            zmean = ymean + (1.0 / sqrt(1.0 + self.norm_v2) - 1.0) * np.dot(
                ymean, self.vn) * self.vn
            self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs) *
                                                     self.mueff) * zmean
            squarenorm = np.sum(self.ps * self.ps)
            self.sigma *= exp(
                (sqrt(squarenorm) / self.chiN - 1) * self.cs / self.ds)
            hsig = squarenorm < (2.0 + 4.0 / (self.N + 1)) * self.N
        elif self.ssa == 'TPA':
            if self.flg_injection:
                alpha_act = np.where(idx == 1)[0][0] - np.where(idx == 0)[0][0]
                alpha_act /= float(self.lam - 1)
                self.ps += self.cs * (alpha_act - self.ps)
                self.sigma *= exp(self.ps / self.ds)
                hsig = self.ps < 0.5
            else:
                self.flg_injection = True
                hsig = True
        else:
            raise NotImplementedError('ssa = ' + self.ssa +
                                      ' is not implemented.')

        # Cumulation
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (
            2 - self.cc) * self.mueff) * np.dot(self.w, sary)

        # Alpha and related variables
        alpha, avec, bsca, invavnn = self._alpha_avec_bsca_invavnn(
            self.vnn, self.norm_v2)
        # Rank-mu
        if self.cmu == 0:
            pvec_mu = np.zeros(self.N)
            qvec_mu = np.zeros(self.N)
        else:
            pvec_mu, qvec_mu = self._pvec_and_qvec(self.vn, self.norm_v2,
                                                   sary / self.dvec, self.w)
        # Rank-one
        if self.cone == 0:
            pvec_one = np.zeros(self.N)
            qvec_one = np.zeros(self.N)
        else:
            pvec_one, qvec_one = self._pvec_and_qvec(self.vn, self.norm_v2,
                                                     self.pc / self.dvec)
        # Add rank-one and rank-mu before computing the natural gradient
        pvec = self.cmu * pvec_mu + hsig * self.cone * pvec_one
        qvec = self.cmu * qvec_mu + hsig * self.cone * qvec_one
        # Natural gradient
        if self.cmu + self.cone > 0:
            ngv, ngd = self._ngv_ngd(self.dvec, self.vn, self.vnn, self.norm_v,
                                     self.norm_v2, alpha, avec, bsca, invavnn,
                                     pvec, qvec)
            # truncation factor to guarantee at most 70 percent change
            upfactor = 1.0
            upfactor = min(upfactor,
                           0.7 * self.norm_v / sqrt(np.dot(ngv, ngv)))
            upfactor = min(upfactor, 0.7 * (self.dvec / np.abs(ngd)).min())
        else:
            ngv = np.zeros(self.N)
            ngd = np.zeros(self.N)
            upfactor = 1.0
        # Update parameters
        self.vvec += upfactor * ngv
        self.dvec += upfactor * ngd

        # update the constants
        self.norm_v2 = np.dot(self.vvec, self.vvec)
        self.norm_v = sqrt(self.norm_v2)
        self.vn = self.vvec / self.norm_v
        self.vnn = self.vn**2


        # Leap
        Cnew = np.ones(self.N)
        for i in range(self.N):
            Cnew[i] = self.dvec[i]**2 * (1 + self.vvec[i]**2)
        self.dig = Cnew

        ci = (self.norm_ci * self.sigma * np.sqrt(self.dig))[self.dim_co:]
        ci_up = self.xmean[self.dim_co:] + ci
        ci_low = self.xmean[self.dim_co:] - ci

        resolution = np.array([bisect_left(self.lim[i], ci_up[i]) - bisect_left(self.lim[i], ci_low[i]) for i in range(self.dim_int)])
        self.xmean[self.dim_co:] = np.array([
            self.xmean[i + self.dim_co] if resolution[i] != 0 else
            self.lim[i][0] - ci[i] if self.xmean[i + self.dim_co] <= self.lim[i][0] else
            self.lim[i][-1] + ci[i] if self.lim[i][-1] < self.xmean[i + self.dim_co] else
            self.lim[i][bisect_left(self.lim[i], self.xmean[i + self.dim_co]) - 1] + ci[i] if self.xmean[i + self.dim_co] <= l_close[i] else
            self.lim[i][bisect_left(self.lim[i], self.xmean[i + self.dim_co])] - ci[i]
            for i in range(self.dim_int)])

    def _check(self):
        if self.fbest < self.ftarget:
            self.is_success = True
            return True, 'ftarget'
        elif self.neval >= self.maxeval:
            return True, 'maxeval'
        else:
            return False, ''

    @staticmethod
    def _alpha_avec_bsca_invavnn(vnn, norm_v2):
        gamma = 1.0 / sqrt(1.0 + norm_v2)
        alpha = sqrt(norm_v2**2 + (1.0 + norm_v2) / max(vnn) * (
            2.0 - gamma)) / (2.0 + norm_v2)
        if alpha < 1.0:  # Compute beta = (1-alpha^2)*norm_v4/(1+norm_v2)
            beta = (4.0 - (2.0 - gamma) / max(vnn)) / (1.0 + 2.0 / norm_v2)**2
        else:
            alpha = 1.0
            beta = 0
        bsca = 2.0 * alpha**2 - beta
        avec = 2.0 - (bsca + 2.0 * alpha**2) * vnn
        invavnn = vnn / avec
        return alpha, avec, bsca, invavnn

    @staticmethod
    def _pvec_and_qvec(vn, norm_v2, y, weights=0):
        y_vn = np.dot(y, vn)
        if isinstance(weights, int) and weights == 0:
            pvec = y**2 - norm_v2 / (1.0 + norm_v2) * (y_vn * (y * vn)) - 1.0
            qvec = y_vn * y - ((y_vn**2 + 1.0 + norm_v2) / 2.0) * vn
        else:
            pvec = np.dot(weights, y**2 - norm_v2 / (1.0 + norm_v2) *
                          (y_vn * (y * vn).T).T - 1.0)
            qvec = np.dot(weights, (y_vn * y.T).T - np.outer(
                (y_vn**2 + 1.0 + norm_v2) / 2.0, vn))
        return pvec, qvec

    @staticmethod
    def _ngv_ngd(dvec, vn, vnn, norm_v, norm_v2, alpha, avec, bsca, invavnn,
                 pvec, qvec):
        rvec = pvec - alpha / (1.0 + norm_v2) * (
            (2.0 + norm_v2) * (qvec * vn) - norm_v2 * np.dot(vn, qvec) * vnn)
        svec = rvec / avec - bsca * np.dot(rvec, invavnn) / (
            1.0 + bsca * np.dot(vnn, invavnn)) * invavnn
        ngv = qvec / norm_v - alpha / norm_v * (
            (2.0 + norm_v2) * (vn * svec) - np.dot(svec, vnn) * vn)
        ngd = dvec * svec
        return ngv, ngd