# utils.py
# basic calculations
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from matplotlib import colors
import pdb
from . import natconst

def calc_lpa(Q, U):
    """ function to calculate polarization angle
    Parameters 
    ----------
    Q : ndarray
    U : ndarray

    Returns
    -------
    ang : ndarray
        the polarization angle in radians and in between 0, 2pi
    """
    ang = np.arctan2(U, Q) / 2.
    reg = ang < 0.
    ang[reg] = ang[reg] + 2*np.pi
    reg = ang > np.pi
    ang[reg] = ang[reg] - np.pi

    return ang

def calc_lpi_rms(Q_rms, U_rms):
    """
    we want to calculate a representative rms for lpi given Q_rms and U_rms
    a simple way is to take the rms of both Q and U together. 
    Realize that the resulting variance is the mean of the variance of the original components
    so we can calulate the rms quite easily
    """
    return np.sqrt(0.5 * (Q_rms**2 + U_rms**2))

def calc_pi_rms(Q_rms, U_rms, V_rms):
    """
    similar to calc_lpi_rms()
    """
    return np.sqrt((Q_rms**2 + U_rms**2 + V_rms**2) / 3)

# ====================
# polarization properties
# ====================
def fn_prob(P, Ptru, rms):
    """
    calculate the probability
    there's probably some overflowing going on in the high S/N limit
    """
    z = (P/rms) * (Ptru/rms)
    return P / rms**2 * special.i0(z) * np.exp(-0.5 * ((P/rms)**2 + (Ptru/rms)**2))

def get_obs_lpi_highSN(Q, U, Q_rms, U_rms):
    """
    calculate the lpi in the high S/N approximation
    """
    lpi_rms = calc_lpi_rms(Q_rms, U_rms)
    return np.sqrt(Q**2 + U**2 - lpi_rms**2)

def get_obs_lpi(Q, U, Q_rms, U_rms, n_per_rms=10, fill_value=np.nan):
    """
    calculate the lpi considering noise
    debias with the maximum probability method
    includes debiasing
    no masking needed for lpi
    simply fix thres to 3, which is the cut off threshold below which we want to debias and above which we will use the high S/N approximation

    Parameters
    ----------
    Q, U : 1d ndarray
    Q_rms, U_rms : float

    n_per_rms : float
        The number of points per sigma for finding the maximum of the probability
    fill_value : 
        obsolete!
    """
    lpi_rms = calc_lpi_rms(Q_rms, U_rms)

    thres = 3

    # raw lpi
    raw = np.sqrt(Q**2 + U**2)

    # first give the lpi in the high S/N limit
    lpi = np.sqrt(Q**2 + U**2 - lpi_rms**2)

    # ==== solve for maximum probability ====
    lowreg = raw <= (thres * lpi_rms)
    measured = raw[lowreg]

    # create an array of true values
    n = np.max([n_per_rms, n_per_rms*thres])
    Ptru = np.linspace(0, thres+1, n) * lpi_rms

    # the probability in 2D
    prob = fn_prob(measured[:,None], Ptru[None,:], lpi_rms)

    # find the index
    inx = np.argmax(prob, axis=1)
    lpi_maxprob = np.array([Ptru[inx[i]] for i in range(len(inx))])

    # replace the regions
    lpi[lowreg] = lpi_maxprob

#    reg = lpi >= lpi_rms
#    lpi[~reg] = fill_value

    return lpi

def get_obs_pi(Q, U, V, Q_rms, U_rms, V_rms, fill_value=np.nan):
    """
    calculate the polarized intensity
    """
    pi_rms = calc_pi_rms(Q_rms, U_rms, V_rms)
    pi = np.sqrt(Q**2 + U**2 + V**2 - pi_rms**2)
    reg = pi >= pi_rms
    pi[~reg] = fill_value
    return pi

def get_obs_lpa(Q, U, Q_rms, U_rms, thres=3, fill_value=np.nan):
    """
    calculate the lpa only at regions with acceptable noise
    """
    lpa = calc_lpa(Q, U)

    lpi_rms = calc_lpi_rms(Q_rms, U_rms)
    lpi = np.sqrt(Q**2 + U**2 - lpi_rms**2)
    reg = lpi >= lpi_rms * thres
    lpa[~reg] = fill_value

    return lpa

def get_obs_lpol(I, Q, U, I_rms, Q_rms, U_rms, thres=3, fill_value=np.nan):
    """
    calculate the linear polarization fraction
    """
    lpi = get_obs_lpi(Q, U, Q_rms, U_rms, fill_value=fill_value)

    lpol = lpi / I

    lpi_rms = calc_lpi_rms(Q_rms, U_rms)

    reg = (I >= I_rms * thres) & (lpi >= lpi_rms * thres)

    lpol[~reg] = fill_value

    return lpol

def get_obs_q(I, Q, I_rms, Q_rms, thres=3, fill_value=np.nan):
    q = Q / I
    reg = (I >= I_rms*thres) & (Q >= Q_rms * thres)
    q[~reg] = fill_value

    pdb.set_trace()

    return q

def get_obs_pol(I, Q, U, V, I_rms, Q_rms, U_rms, V_rms, thres=3, fill_value=np.nan):
    """
    calculate the polarization fraction 
    """
    pi = get_obs_pi(Q, U, V, Q_rms, U_rms, V_rms, fill_value=fill_value)

    pol = pi / I
    pi_rms = calc_pi_rms(Q_rms, U_rms, V_rms)

    reg = (I >= I_rms * thres) & (pi >= pi_rms*thres)

    pol[~reg] = fill_value
    return pol

def get_obs_cpol(I, V, I_rms, V_rms, thres=3, fill_value=np.nan):
    """
    calculate the circular polarization fraction 
    """
    cpol = V / I
    reg = (I >= I_rms * thres) & (V >= V_rms * thres)
    cpol[~reg] = fill_value
    return cpol

def get_polarization(I, Q, U, V, I_rms, Q_rms, U_rms, V_rms, thres=3, fill_value=0): 
    """ 
    A convenience function to get all the important polarization in one go

    Parameters
    ----------
    I, Q, U, V : ndarray
    I_rms, Q_rms, U_rms, V_rms : float
    """
    out = {}

    # lpi
    out['lpi'] = get_obs_lpi(Q, U, Q_rms, U_rms, fill_value=fill_value)

    # total polarized intensity
    out['pi'] = get_obs_pi(Q, U, V, Q_rms, U_rms, V_rms, fill_value=fill_value)

    # polarization angle
    out['lpa'] = get_obs_lpa(Q, U, Q_rms, U_rms, thres=thres, fill_value=fill_value)

    # linear polarization fraction
    out['lpol'] = get_obs_lpol(I, Q, U, I_rms, Q_rms, U_rms, thres=thres, fill_value=fill_value)

    # total polarization fraction
    out['pol'] = get_obs_pol(I, Q, U, V, I_rms, Q_rms, U_rms, V_rms, thres=thres, fill_value=fill_value)

    # circular polarization
    out['cpol'] = get_obs_cpol(I, V, I_rms, V_rms, thres=thres, fill_value=fill_value)

    return out

def fn_planck(nu, T):
    """ black body radiation 
    """
    x = natconst.hh * nu / natconst.kk / T
    return 2 * natconst.hh * nu**3 / natconst.cc**2 / ( np.exp(x) - 1. )

def pp1d_intensity(tau, mu, T, nu, alb):
    """ solve the intensity from plane-parallel solution
    """
    eps = np.sqrt(1. - alb)
    rt3 = np.sqrt(3)
    a = 1./( np.exp(-rt3*eps*tau) * (eps-1.) - (eps+1.) )
    b = (1. - np.exp(-(rt3*eps+1./mu)*tau)) / (rt3*eps*mu+1.)
    c = ( np.exp(-tau/mu) - np.exp(-rt3*eps*tau) ) / (rt3*eps*mu - 1.)

    f = a * (b + c)

    Bnu = fn_planck(nu, T)

    return Bnu * ( (1. - np.exp(-tau/mu)) + alb * f )


