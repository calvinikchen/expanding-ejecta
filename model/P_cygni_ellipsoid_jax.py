"""
P Cygni Ellipsoid Module

This module implements the P_cygni_ellipsoid class, which calculates the spatially-dependent
absorption and emission profile of the P Cygni line profile using Sobolev's approximation
for an ellipsoidal geometry.

The module uses JAX for efficient numerical computations and includes methods for
calculating velocities, geometric dilution factors, and optical depths.

Author: I-Kai Chen
Date: September 30, 2024
"""
import os
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const
import pandas as pd

"""
# Load data from CSV file
table = pd.read_csv('model/table/w_table.csv')
data = jnp.array(table['0'].to_numpy().reshape((100, 100, 90)))

# Define parameter ranges
eta = jnp.linspace(-1, 1, 100)
r = jnp.linspace(-1, 1, 100)
theta = jnp.linspace(0, np.pi/2, 90)
"""


class P_cygni_ellipsoid(eqx.Module):
    """
    Calculate the spatially-dependent absorption and emission profile of the P Cygni line profile
    using Sobolev's approximation for an ellipsoidal geometry.

    This class models the P Cygni profile for an ellipsoidal geometry, taking into account various
    physical parameters and geometric factors to compute the absorption and emission components
    of the spectral line profile.

    Attributes:
        nu_obs (jax.Array): Observed photon frequency array.
        th_obs (jax.Array): Observed angular coordinate array (0 at the center of the source).
        ang_obs (jax.Array): Angular observation array.
        v0 (jnp.float64): Ejecta velocity at the photosphere in m/s.
        nu0 (jnp.float64): Rest frequency of the line.
        ds (jnp.float64): Distance to the source in m.
        t (jnp.float64): Time since SN explosion in s.
        th_max (jnp.float64): Angular coordinate of the edge of the photosphere assuming homologous expansion.
        eta (jnp.float64): Ellipsoid shape parameter.
        theta (jnp.float64): Ellipsoid orientation angle.
        x (jax.Array): x-coordinate of the ejecta.
        y (jax.Array): y-coordinate of the ejecta.
        z (jax.Array): z-coordinate of the ejecta.
        r (jax.Array): Radial coordinate of the ejecta.
        rho_sq (jax.Array): Squared cylindrical radius.
        vel (jax.Array): Velocity of the ejecta where the Sobolev optical depth is non-zero.
        geo_w (jax.Array): Geometric dilution factor at non-zero Sobolev optical depth.
        tau (jax.Array): Sobolev optical depth.
    """

    nu_obs: jax.Array
    th_obs: jax.Array
    ang_obs: jax.Array
    v0: jnp.float64
    nu0: jnp.float64
    ds: jnp.float64
    t: jnp.float64
    th_max: jnp.float64
    eta: jnp.float64
    theta: jnp.float64
    x: jax.Array
    y: jax.Array
    z: jax.Array
    r: jax.Array
    rho_sq: jax.Array
    vel: jax.Array
    geo_w: jax.Array
    tau: jax.Array

    # Load data from CSV file
    table = pd.read_csv('model/table/w_table.csv')
    data = jnp.array(table['0'].to_numpy().reshape((100, 100, 90)))


    def __init__(self, nu_obs, th_obs, ang_obs=jnp.arange(0, np.pi, np.pi/10),
                 v0=8e6, nu0=1, ds=const.parsec*1e6, t=10*const.day, eta=1.2, theta=0):
        """
        Initialize the P_cygni_ellipsoid model.

        Args:
            nu_obs (array-like): Observed photon frequency array.
            th_obs (array-like): Observed angular coordinate array.
            ang_obs (array-like, optional): Angular observation array. Defaults to 10 evenly spaced angles from 0 to π.
            v0 (float, optional): Ejecta velocity at the photosphere in m/s. Defaults to 8e6.
            nu0 (float, optional): Rest frequency of the line. Defaults to 1.
            ds (float, optional): Distance to the source in m. Defaults to 1 Mpc.
            t (float, optional): Time since SN explosion in s. Defaults to 10 days.
            eta (float, optional): Ellipsoid shape parameter. Defaults to 1.2.
            theta (float, optional): Ellipsoid orientation angle. Defaults to 0.
        """
        self.nu_obs = jnp.expand_dims(nu_obs, axis=(0, 1))
        self.th_obs = jnp.expand_dims(th_obs, axis=(1, 2))
        self.ang_obs = jnp.expand_dims(ang_obs, axis=(0, 2))
        self.v0 = v0
        self.nu0 = nu0
        self.ds = ds
        self.t = t
        self.th_max = v0 * t / ds
        self.eta = eta
        self.theta = theta

        # Calculate Cartesian coordinates
        x = self.th_obs * jnp.cos(self.ang_obs)
        y = self.th_obs * jnp.sin(self.ang_obs)

        self.x = x / self.th_max
        self.y = y / self.th_max
        self.z = (1 - self.nu0 / self.nu_obs) * const.c * self.t / self.ds / self.th_max

        # Calculate radial coordinate
        self.r = jnp.sqrt(self.x**2 + self.y**2 + self.z**2)

        # Calculate squared cylindrical radius
        self.rho_sq = x**2 / (self.eta**2 * jnp.cos(theta)**2 + jnp.sin(theta)**2) + y**2

        # Calculate velocity and geometric dilution factor
        self.vel = self._get_v()
        self.geo_w = self._get_geo_W()

        # Initialize tau as None (to be calculated later)
        self.tau = None


    def _get_v(self):
        """
        Calculate the ejecta velocity for given frequency and angular position.

        Returns:
            jax.Array: Array of velocities corresponding to the given frequency and angular position.
        """
        vel = jnp.sqrt(
            self.x**2 * (jnp.cos(self.theta)**2 / self.eta**2 + jnp.sin(self.theta)**2) +
            self.y**2 +
            self.z**2 * (jnp.sin(self.theta)**2 / self.eta**2 + jnp.cos(self.theta)**2) +
            self.x * self.z * (1 / self.eta**2 - 1) * jnp.sin(2 * self.theta)
        ) * self.th_max * self.ds / self.t * (2 * ((1 - self.nu0 / self.nu_obs) >= 0) - 1)

        # Apply conditions for physical regions
        cond1 = jnp.logical_or((vel >= self.v0), jnp.isclose(vel / self.v0, 1))
        cond2 = jnp.logical_or(jnp.isclose(self.rho_sq / self.th_max**2, 1), (self.rho_sq >= self.th_max**2))
        cond = jnp.logical_or(cond1, cond2)

        return vel * cond


    def _get_geo_W(self):
        """
        Calculate the geometric dilution factor.

        Returns:
            jax.Array: Array of dilution factors corresponding to the given frequency and angular position.
        """
        new_x = self.x * jnp.cos(self.theta) + self.z * jnp.sin(self.theta)
        x = new_x / self.r

        # Polynomial approximation for arcsin
        arg = 0.99954 - 0.454064*x**2 - 0.844813*x**4 + 3.96537*x**6 - 10.0582*x**8 + 11.3462*x**10 - 4.86265*x**12
        ang = 1.01044*arg - 0.042046*arg**3 + 1.15962*arg**5 - 2.00355*arg**7 + 1.33027*arg**9

        shape = ang.shape
        r = jnp.ravel(jnp.log10(self.r)) * 99/2 + 99/2
        ang = jnp.abs(jnp.ravel(ang)) / np.pi * 2 * 89
        eta = jnp.zeros(len(r)) + jnp.log10(self.eta) * 99/2 + 99/2

        # Interpolate geometric dilution factor
        w = jax.scipy.ndimage.map_coordinates(P_cygni_ellipsoid.data,
                                              jnp.array([eta, r, ang]), order=1)
        w = w.reshape(shape)

        return jnp.nan_to_num(w, nan=0)


    def opt_depth(self, tau0=2., n=5.):
        """
        Calculate the Sobolev optical depth.

        This method assumes a power law for optical depth: tau = tau0 * (v/v0)^(-n)

        Args:
            tau0 (float, optional): Sobolev optical depth at the photosphere. Defaults to 2.
            n (float, optional): Spectral index of the power law for optical depth as a function of ejecta velocity. Defaults to 5.
        """
        self.vel = self.vel.astype('complex128')
        tau = jnp.abs(tau0 * (self.vel / self.v0 + (self.vel == 0))**(-n) * (self.vel != 0))
        self.tau = jnp.nan_to_num(tau, nan=0)


    def absorption(self):
        """
        Calculate the absorption profile of the P Cygni line profile.

        Precondition: self.opt_depth() must be called before this method.
        """
        assert self.tau is not None, "Optical depth not calculated yet. Please run P_cygni_ellipsoid.opt_depth() first."

        self.pcyg_abs = jnp.exp(-self.tau) * (self.rho_sq < self.th_max**2)


    def emission(self):
        """
        Calculate the emission profile of the P Cygni line profile.

        Precondition: self.opt_depth() must be called before this method.
        """
        assert self.tau is not None, "Optical depth not calculated yet. Please run P_cygni_ellipsoid.opt_depth() first."

        self.pcyg_em = (1 - jnp.exp(-self.tau)) * self.geo_w
