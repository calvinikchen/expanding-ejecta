import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
#import healpy as hp
import scipy.constants as const
import scipy.optimize as op
import scipy.integrate as integrate
import matplotlib as mpl
import P_cygni as pc
from black_body import b_nu, b_lam
import scipy.fft as fft
import time
import scipy.interpolate as interp

class Correlator_I_Hankel(pc.P_cygni):
    """
    Class used to calculate the fractional intensity correlation and noise
    of the SN P Cygni line profile as seen by an intensity intereferometer.

    This is a subclass of P_cygni

    Attributes
    ----------
    c_norm : np.array
        normalization for the fractional intensity correlator C_k. Formula
        given in overleaf: sqrt(1 + 2*(sigma_k*sigma_t)^2)
    n_nu : int
        number of spectral channel
    n_grid : int
        number of 1D grid for calculating the fast fourier transform
    d_gid : float
        angular grid spacing
    area : float
        telescope collecting area [m^2]
    eta : float
        telescope cellecting efficiency
    spectra_r : float
        spectral resolution of the telescope
    sigma_t : float
        temporal resolution of the telescope [s]
    c_k : np.array
        2D fractional intensity correlator
    sigma_ck : np.array
        uncertainty in the fractional intensity correlator, formula given
        in overleaf
    _spec_2d : np.array
        2D relative flux of the P Cygni profile
    """


    def __init__(self, nu_obs, r_ar,
                 v0 = 8e6, nu0 = 1, ds = const.parsec*1e6, t = 10*const.day,
                 area = 25*np.pi, eta = 0.5, spectra_r = 1e4, sigma_t = 1e-11,
                 tau0 = 2., n = 5., em = True):
        """
        Class constructor, sets the basic parameters for the simplified SN model
        and the telescope parameter

        Parameters
        ----------
        nu_obs : np.array
            the array of the observed photon frequency
        x_ar : np.array
            the 1D array of the grid to calculate the fast fourier transform
        v0 : float
            the ejecta velocity at the photosphere in [m/s]
            (default: 8e6)
        nu0 : float
            the rest frequency of the line
            (default: 1)
        ds : float
            distance to the source in [m]
            (default: 1 Mpc)
        t : float
            time since SN explosion in [s]
            (default: 10 days)
        area : float
            telescope collecting area [m^2]
            (default: 25*pi)
        eta : float
            telescope cellecting efficiency
            (default: 0.5)
        spectra_r : float
            spectral resolution of the telescope
        sigma_t : float
            temporal resolution of the telescope [s]
        tau0 : float
            optical depth at the photosphere
        n : float
            spectral index of the optical depth profile
        """
        # creates the full grid for the P_cygni class
        #r_grid = np.logspace(-3, 1, 1024)


        # calls the parent class constructor
        super().__init__(nu_obs, r_ar, v0, nu0, ds, t)

        self.r_ar = r_ar

        self.c_norm = np.sqrt(1 + 2*(sigma_t*nu_obs/spectra_r*2*np.pi)**2)

        self.n_nu = len(nu_obs)
        self.n_grid = len(r_ar)

        dln = np.log(self.r_ar[1]/self.r_ar[0])
        self.d_grid = dln

        self.k_ar = 1/self.r_ar[::-1]

        self.area = area
        self.eta = eta
        self.spectra_r = spectra_r
        self.sigma_t = sigma_t

        # calculate the optical depth
        self.opt_depth(tau0, n)

        # calculate the line profile
        self.absorption()
        self.emission()

        # calculate the fractional intensity correlator
        self._fourier(em)


    def ck_err(self, t_obs = 3600., temp = 6000.):
        """
        method to calculate the ucertainty in C_k given observation time and
        temperature of the black body

        Parameters
        ----------
        t_obs : float
            total observation (exposure) time [s]
            (default: 3600)
        temp : float
            temperature of the black body background [K]
            (default: 6000)
        """
        sig_ck_sq = 1/np.sqrt(4*np.pi)/self._phot_N(temp)**2/self.sigma_t/t_obs
        self.sigma_ck = np.sqrt(sig_ck_sq).reshape(self.n_nu)


    def interp_ck(self, nu_ar, kd_ar):
        """
        method that returns the C(k) interpolation of spectral channel nu_ar
        and Fourier coordinate kd_ar

        Parameters
        ----------
        nu_ar : np.array
            the spectral channel to perform the interpolation on
        kd_ar : np.array
            the Fourier coordinate to perform the interpolation on

        Returns
        -------
        a 2D array of the corresponding C(k) from interpolation
        """
        x = self.nu_obs.flatten()
        y = self.k_ar/np.pi
        #print(y)

        points = (x, y)

        #print(y/self.th_max)

        p = interp.interpn(points, np.log(self.c_k),
                           np.array([nu_ar, kd_ar]).T, method = 'splinef2d')
        return p


    def interp_ck_err(self, nu_ar):
        """
        method that returns the uncertinty in C(k) interpolation of
        spectral channel nu_ar

        Parameters
        ----------
        nu_ar : np.array
            the spectral channel to perform the interpolation on

        Returns
        -------
        a 1D array of the corresponding uncertainty of C(k) from interpolation
        """
        x = self.nu_obs.flatten()

        f = interp.interp1d(x, self.sigma_ck)
        err = f(nu_ar)

        return err


    def interp_ck_norm(self, nu_ar):
        """
        method that returns the nomralization in C(k) interpolation of
        spectral channel nu_ar

        Parameters
        ----------
        nu_ar : np.array
            the spectral channel to perform the interpolation on

        Returns
        -------
        a 1D array of the corresponding normalization of C(k) from interpolation
        """
        x = self.nu_obs.flatten()

        f_c = interp.interp1d(x, self.c_norm)
        c_norm = f_c(nu_ar)

        return c_norm


    def _fourier(self, em):
        """
        helper method to calculate the fractional intensity correlator
        simply using fft2 to perfoeem the fast fourier tranform
        """
        if em:
            spec = (self.pcyg_em + self.pcyg_abs).T
        else:
            spec = (self.pcyg_abs).T

        # stores 2d intensity for calculating uncertainty in C_k
        self._spec = spec

        f_k = fft.fht(spec*self.r_ar, self.d_grid, mu=0)
        f_k = f_k/self.k_ar

        norm = np.sum(self._spec*self.r_ar**2*self.d_grid, axis = 1)
        self.norm = norm

        c_k = (f_k/np.expand_dims(norm, axis = 1))**2/np.expand_dims(self.c_norm, axis = 1)
        self.c_k = c_k


    def _phot_N(self, temp):
        """
        helper method to calculate the total photon number in each spectral channel
        assuming a black body background

        Parameters
        ----------
        temp : float
            temperature of the black body [K]

        Returns
        -------
        Number of photon collected in each spectral channel with an exposure time of 1s
        """
        spec_int = self.norm*2*np.pi#*self.th_max**2
        #print(spec_int)
        return b_nu(self.nu_obs, temp)*spec_int/const.h*self.area*self.eta/self.spectra_r*np.sqrt(2*np.pi)
