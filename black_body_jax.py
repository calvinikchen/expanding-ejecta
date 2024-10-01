"""
Black Body Radiance Module

This module provides a function to calculate the logarithm of the spectral radiance
of a black body using the Planck function. It utilizes JAX for efficient numerical
computations.

The module includes:
- b_nu: Function to compute the logarithm of spectral radiance

Author: I-Kai Chen
Date: October 1, 2024
"""

import jax
import jax.numpy as jnp
import scipy.constants as const

def b_nu(nu, temp):
    """
    Calculate the logarithm of the spectral radiance of a black body.

    This function computes the logarithm of the spectral radiance (B_ν) of a black body
    using the Planck function. It uses JAX for efficient numerical computations.

    Args:
        nu (jax.Array): Frequency in Hz
        temp (float): Temperature of the black body in Kelvin

    Returns:
        jax.Array: Logarithm of the spectral radiance in W⋅sr^−1⋅m^−2⋅Hz^−1

    Note:
        This function returns the natural logarithm of B_ν to avoid numerical overflow
        for high frequencies or low temperatures.
    """
    # Calculate the logarithm of the Planck function
    return (jnp.log(2*const.h) + 3*jnp.log(nu) - 2*jnp.log(const.c) -
            jnp.log(jnp.exp(const.h*nu/const.k/temp) - 1))
