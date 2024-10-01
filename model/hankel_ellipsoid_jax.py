"""
Correlator Ellipsoid Module

This module implements the Correlator_ellipsoid class, which extends the P_cygni_ellipsoid
class to calculate intensity correlators in ellipsoidal P Cygni profiles. It provides methods for
calculating multipole moments and Fourier transforms of the profile, as well as error
estimation for the correlation function.

The module uses JAX for efficient numerical computations and includes several helper
functions for specialized calculations.

Author: I-Kai Chen
Date: September 30, 2024
"""
import sys
sys.path.append('model/')

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as const
from P_cygni_ellipsoid_jax import P_cygni_ellipsoid
from black_body_jax import b_nu
import jax
import jax.numpy as jnp


class Correlator_ellipsoid(P_cygni_ellipsoid):
    """
    A class for calculating intensity correlators around a P Cygni profile of an
    ellipsoidal supernova.

    This class extends the P_cygni_ellipsoid class and adds methods for
    calculating multipole moments and Hankel transforms of the image of the
    supernova.

    This class is JAX compatible with jax.grad ready for potential speed up
    using faster MCMC methods such as Hamiltonian Monte Carlo.

    Attributes:
        r_ar (jax.Array): Array of radii
        c_norm (jax.Array): Normalization factor for correlation function
        n_nu (int): Number of frequency points
        n_grid (int): Number of grid points
        d_grid (jnp.float64): Grid spacing
        n_ang (jnp.float64): Number of angular points
        k_ar (jax.Array): Array of wavenumbers
        area (jnp.float64): Area of the observing telescope
        eff (jnp.float64): Efficiency of the observing telescope
        spectra_r (jnp.float64): Spectral resolution
        sigma_t (jnp.float64): Timing resolution
        ss (jax.Array): Sum of absorption and emission profiles
        spec (jax.Array): Spectrum
        pcyg_abs (jax.Array): P Cygni absorption profile
        pcyg_em (jax.Array): P Cygni emission profile
        f_m (jax.Array): Multipole moments
        t (jax.Array): Time array for Fourier transform
        dt (jnp.float64): Time step
        f_k (jax.Array): Fourier transform of multipole moments
    """

    r_ar: jax.Array
    c_norm: jax.Array
    n_nu: int
    n_grid: int
    d_grid: jnp.float64
    n_ang: jnp.float64
    k_ar: jax.Array
    area: jnp.float64
    eff: jnp.float64
    spectra_r: jnp.float64
    sigma_t: jnp.float64
    ss: jax.Array
    spec: jax.Array
    pcyg_abs: jax.Array
    pcyg_em: jax.Array
    f_m: jax.Array
    t: jax.Array
    dt: jnp.float64
    f_k: jax.Array


    def __init__(self, nu_obs, r_ar, ang_obs=jnp.linspace(0, np.pi, 11),
                 v0=8e6, nu0=1, ds=const.parsec*1e6, t=10*const.day, eta=1.2,
                 theta=0, tau0=2, n=5, area=25*jnp.pi, eff=0.5, spectra_r=1e4,
                 sigma_t=1e-11):
        """
        Initialize the Correlator_ellipsoid object.

        Args:
            nu_obs (array-like): Observed frequencies
            r_ar (array-like): Array of radii
            ang_obs (array-like, optional): Array of observation angles. Defaults to 11 angles from 0 to pi.
            v0 (float, optional): Ejecta velocity at the photosphere in m/s. Defaults to 8e6.
            nu0 (float, optional): Rest frequency of the line. Defaults to 1.
            ds (float, optional): Distance to the source in m. Defaults to 1 Mpc.
            t (float, optional): Time since SN explosion in s. Defaults to 10 days.
            eta (float, optional): Ellipsoid shape parameter. Defaults to 1.2.
            theta (float, optional): Ellipsoid orientation angle. Defaults to 0.
            tau0 (float, optional): Optical depth at the photosphere. Defaults to 2.
            n (float, optional): Power-law index for optical depth. Defaults to 5.
            area (float, optional): Area of the observing telescope. Defaults to 25π.
            eff (float, optional): Efficiency of the observing telescope. Defaults to 0.5.
            spectra_r (float, optional): Spectral resolution. Defaults to 1e4.
            sigma_t (float, optional): Timing resolution. Defaults to 1e-11.
        """
        super().__init__(nu_obs, r_ar, ang_obs, v0, nu0, ds, t, eta, theta)

        self.r_ar = r_ar
        self.c_norm = jnp.sqrt(1 + 2*(sigma_t*nu_obs/spectra_r*2*np.pi)**2)

        self.n_nu = len(nu_obs)
        self.n_grid = len(r_ar)

        dln = jnp.log(self.r_ar[1]/self.r_ar[0])
        self.d_grid = dln
        self.n_ang = len(ang_obs)

        self.k_ar = 1/self.r_ar[::-1]

        self.area = area
        self.eff = eff
        self.spectra_r = spectra_r
        self.sigma_t = sigma_t

        # Calculate the optical depth profile
        self.opt_depth(tau0, n)

        # Calculate the line profile
        self.absorption()
        self.emission()

        r = r_ar/self.th_max
        self.ss = self.pcyg_abs + self.pcyg_em
        norm = jnp.sum((r<1)*r**2)*(self.n_ang-1)
        self.spec = jnp.sum(self.ss[:,:-1,:]*jnp.expand_dims(r**2, axis=(1,2)),
                            axis=(0,1))/norm

        self.multipole()

        t = jnp.fft.fftfreq(self.n_grid, self.d_grid)*2*jnp.pi
        self.t = t
        self.dt = t[1] - t[0]
        self.fourier()


    def multipole(self):
        """
        Calculate multipole moments of the profile.
        """
        f_m = v_m(jnp.arange(8), self)
        self.f_m = f_m


    def fourier(self):
        """
        Perform Fourier transform of the multipole moments.
        """
        norm = jnp.sum(self.f_m[0]*jnp.expand_dims((self.r_ar/self.th_max)**2,
                                                    axis=1)*self.d_grid*jnp.exp(-self.d_grid), axis=0)

        # Calculate Hankel transforms for multipole moments 0 to 7
        f0 = self.hankel(0)
        f1 = self.hankel(1)
        f2 = self.hankel(2)
        f3 = self.hankel(3)
        f4 = self.hankel(4)
        f5 = self.hankel(5)
        f6 = self.hankel(6)
        f7 = self.hankel(7)
        f_k = jnp.array((f0,f1,f2,f3,f4,f5,f6,f7))

        sum_f = (jnp.sum(jnp.expand_dims((1j*jnp.exp(-1j*self.ang_obs[0,:,:]))**(-jnp.arange(8)), axis=(2,3))*jnp.expand_dims(f_k, axis=0), axis=1) +
                 jnp.sum(jnp.expand_dims((-1j*jnp.exp(-1j*self.ang_obs[0,:,:]))**(-jnp.arange(-1,-8,-1)), axis=(2,3))*jnp.expand_dims(f_k[1:], axis=0), axis=1))

        self.f_k = jnp.abs(sum_f/jnp.expand_dims(norm, axis=(0,1)))**2/jnp.expand_dims(self.c_norm, axis=(0,1))


    def hankel(self, mu):
        """
        Perform Hankel transform for a given order mu.

        Args:
            mu (int): Order of the Hankel transform

        Returns:
            jax.Array: Hankel transform result
        """
        t = jnp.fft.fftfreq(self.n_grid, self.d_grid)*2*jnp.pi
        dt = t[1] - t[0]
        f_k = jnp.fft.fft(jnp.expand_dims(q_fast(t,mu), axis=1)*jnp.fft.fft(self.f_m[mu]*jnp.expand_dims((self.r_ar/self.th_max)**(mu+1), axis=1
            ), axis=0), axis=0)*jnp.expand_dims((self.k_ar*self.th_max)**(mu-1)*jnp.exp(-(self.k_ar*self.th_max*mu/1e2)**2), axis=1)*dt*self.d_grid/2/jnp.pi
        f_k = jnp.roll(f_k,-1,axis=0)
        f_k = f_k.at[-1,:].set(f_k[-2,:])

        return f_k


    def ck_err(self, t_obs=3600., temp=6000.):
        """
        Calculate the uncertainty in C_k given observation time and temperature of the black body.

        Args:
            t_obs (float, optional): Total observation (exposure) time in seconds. Defaults to 3600.
            temp (float, optional): Temperature of the black body background in Kelvin. Defaults to 6000.
        """
        sig_ck_sq = 1/jnp.sqrt(4*jnp.pi)/self._phot_N(temp)**2/self.sigma_t/t_obs
        self.sigma_ck = jnp.sqrt(sig_ck_sq).reshape(self.n_nu)


    def _phot_N(self, temp):
        """
        Calculate the total photon number in each spectral channel assuming a black body background.

        Args:
            temp (float): Temperature of the black body in Kelvin

        Returns:
            jax.Array: Number of photons collected in each spectral channel with an exposure time of 1s
        """
        pn_log = (b_nu(self.nu_obs, temp) + jnp.log(self.spec) - jnp.log(const.h) +
                  jnp.log(self.area*self.eff/self.spectra_r*jnp.sqrt(2*jnp.pi)) +
                  2*jnp.log(self.th_max))

        return jnp.exp(pn_log).flatten()

#-----------------
# Helper functions

def multipole_helper(i, ss):
    """
    Helper function for multipole calculation.

    Args:
        i (int): Multipole order
        ss (Correlator_ellipsoid): Correlator_ellipsoid object

    Returns:
        jax.Array: Power in the ith multipole order of the image of a Correlator_ellipsoid object
    """
    return (jnp.sum(ss.ss*jnp.cos(ss.ang_obs*i), axis=1) - ss.ss[:,-1,:]*((i+1)%2))/(ss.n_ang - (i+1)%2)


def phi1(t):
    """
    Calculate the phi1 function used in q_fast.

    Args:
        t (jax.Array): Input array

    Returns:
        jax.Array: Calculated phi1 values
    """
    n = 10
    m_ar = jnp.arange(1,n+1)
    r = jnp.sqrt((n + .5)**2 + t**2)
    phi = jnp.arctan(2*t/(2*n + 1))
    tmp = jnp.sum(jnp.arctan(2*jnp.expand_dims(t, axis=0)/(2*jnp.expand_dims(m_ar, axis=1) - 1)), axis=0)

    return tmp - t*jnp.log(r) + t - n*phi + jnp.sin(phi)/12/r - jnp.sin(3*phi)/360/r**3 + jnp.sin(5*phi)/1260/r**5


def q_fast(t, mu):
    """
    Fast calculation of the q function used in the Hankel transform.

    Args:
        t (jax.Array): Input array
        mu (int): Order of the q function

    Returns:
        jax.Array: Calculated q function values
    """
    ang = 2*phi1(-t/2)
    factor = (-1-1j*t)/jnp.prod((jnp.expand_dims(jnp.arange(-1.,2.*mu,2.), axis=0)-1j*jnp.expand_dims(t,axis=1)), axis=1)

    return 2**(1j*t)*jnp.exp(1j*ang)*factor


# Vectorized multipole helper function
v_m = jax.vmap(multipole_helper, (0, None))
