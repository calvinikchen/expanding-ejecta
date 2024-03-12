import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
import healpy as hp
import scipy.constants as const
import scipy.optimize as op
import scipy.integrate as integrate
import matplotlib as mpl


class P_cygni(object):
    """
    Class used to calculate the spatially-dependent abosrption and emission
    profile of the P Cygni line profile using Sobolev's approximation

    Attributes
    ----------
    nu_obs : np.array
        the array of the observed photon frequency
    th_obs : np.array
        the array of the observed angular coordinate. 0 at the center of the source
    v0 : float
        the ejecta velocity at the photosphere in [m/s]
    nu0 : float
        the rest frequency of the line
    ds : float
        distance to the source in [m]
    t : float
        time since SN explosion in [s]
    th_max : float
        angular coordinate of the edge of the photosphere assuming homologous expansion
        th_max = v0*t/ds
    vel : np.array
        given the observed frequency and angular location, the corresponding velocity of
        the ejecta where the red/blue-shifted frequency gives a non-zero Sobolev optical depth
    geo_W : np.array
        given the observed frequency and angular location, the geometric dillution factor
        at non-zero Sobolev optical depth for calculating the line source function
    tau : np.array
        given the observed frequency, angular location, and a model for the optical depth
        profile, the corresponding Sobolev optical depth
    pcyg_abs : np.array
        the absorption part of the P Cygni line profile calculated with the parameters above
    pcyg_em : np.array
        the emission part of the P Cygni line profile calculated with the parameters above
    """


    def __init__(self, nu_obs, th_obs, v0 = 8e6, nu0 = 1, ds = const.parsec*1e6, t = 10*const.day):
        """
        Class constructor, sets the basic parameters for the simplified SN model

        Parameters
        ----------
        nu_obs : np.array
            the array of the observed photon frequency
        th_obs : np.array
            the array of the observed angular coordinate. 0 at the center of the source
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
        """
        self.nu_obs = np.expand_dims(nu_obs, axis = 0)
        self.th_obs = np.expand_dims(th_obs, axis = 1)
        self.v0 = v0
        self.nu0 = nu0
        self.ds = ds
        self.t = t
        self.th_max = v0*t/ds
        self.vel = self._get_v()
        self.geo_w = self._get_geo_W()
        self.tau = None


    def _get_v(self):
        """
        a helper method to calculate the ejecta velocity when the given frequency is equal to the
        rest frequency at the rest frame of the ejecta, the velocity will be zero for the unphysical
        regions (i.e. inside the photosphere or behind the occulted region).
        Used in class constructor

        Returns
        -------
        array of velocities correponds to the given frequency and angular position
        """
        vel = np.sqrt((self.ds*self.th_obs/self.t)**2 +
                      (1- self.nu0/self.nu_obs)**2*const.c**2)*(2*((1- self.nu0/self.nu_obs)>=0) - 1)

        # homologous expansion has velocity > ejecta velocity at photosphere
        # OR the ejecta can move backward along the LOS only above the photosphere
        cond1 = np.logical_or((vel >= self.v0) , np.isclose(vel/self.v0, 1))
        cond2 = np.logical_or(np.isclose(self.th_obs/self.th_max, 1), (self.th_obs >= self.th_max))
        cond = np.logical_or(cond1 , cond2)

        return vel*cond


    def _get_geo_W(self):
        """
        a helper method to calculate the geometric dilution factor at the position when
        the given frequency is equal to the rest frequency at the rest frame of the ejecta.
        The velocity will be zero for the unphysical regions
        (i.e. inside the photosphere or behind the occulted region).
        Used in class constructor

        Returns
        -------
        array of dilution factors correponds to the given frequency and angular position
        """
        w = 1/2*(1-np.sqrt(np.abs(1-(self.v0/self.vel)**2)))*(self.vel!=0)

        return np.nan_to_num(w, nan = 0)


    def opt_depth(self, tau0 = 2., n = 5.):
        """
        a method to calculate the Sobolev optical depth at the position when the given
        frequency is equal to the rest frequency at the rest frame of the ejecta,
        the optical depth will be zero for the unphysical regions
        (i.e. inside the photosphere or behind the occulted region)
        Stores the value to class attribute self.tau.

        This assums a power law for optical depth
        tau = tau0*(v/v0)^(-n)

        Parameters
        ----------
        tau0 : float
            the Sobolev optical depth at the photosphere
        n : float
            spectral indeex of the power law for the optical depth as a function of
            ejecta velocity
        """
        self.vel = self.vel.astype('complex')
        tau = np.abs(tau0*(self.v0/self.vel)**(n)*(self.vel!=0))
        self.tau = np.nan_to_num(tau, nan = 0)


    def absorption(self):
        """
        a method to calculate the absorption profile of the P Cygni line profile

        Precondition
        ------------
        self.opt_depth() has been run to calculate the optical depth
        so self.tau is not None
        """
        assert not (self.tau is None), ("Optical depth not calculated yet, " +
                                      "please run P_cygni.opt_depth() first")

        self.pcyg_abs = np.exp(-self.tau)*(self.th_obs < self.th_max)


    def emission(self):
        """
        a method to calculate the emission profile of the P Cygni line profile

        Precondition
        ------------
        self.opt_depth() has been run to calculate the optical depth
        so self.tau is not None
        """
        assert not (self.tau is None), ("Optical depth not calculated yet, " +
                                      "please run P_cygni.opt_depth() first")

        self.pcyg_em = (1 - np.exp(-self.tau))*self.geo_w
