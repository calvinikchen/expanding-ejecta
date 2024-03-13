import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
import healpy as hp
import scipy.constants as const
import scipy.optimize as op
import scipy.integrate as integrate
import matplotlib as mpl
from P_cygni import P_cygni



def b_nu(nu, temp):
    return 2*const.h*nu**3/const.c**2/(np.exp(const.h*nu/const.k/temp)-1)

def b_lam(lam, temp):
    return 2*const.h*const.c**2/lam**5/(np.exp(const.h*const.c/const.k/temp/lam)-1)


def flux_nu(nu, temp = 6000, r_sn = 1e13, d_sn = 1e6*const.parsec):
    return (r_sn/d_sn)**2*b_nu(nu, temp)*1e26

def flux_lam(nu, temp = 6000, r_sn = 1e13, d_sn = 1e6*const.parsec):
    return (r_sn/d_sn)**2*b_lam(nu, temp)
