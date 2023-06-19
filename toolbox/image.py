"""
file to include classes related to polarization images from observations

The units of the image grid is in arcseconds. 
North is the first axis and East is the second axis, since this is the typical way of thinking about x and y axes. The Stokes parameters are defined such that x and y are along North and East respectively, following the IAU definition. 

There are two classes: 
- imageCut
- intensity
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pdb
import copy
import subprocess
from astropy.io import fits
from scipy import signal
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.ndimage import map_coordinates
from . import natconst, utils

# ====
# cut object
# ====
class imageCut(object):
    """ an object to record the info of 
    a spatial cut through an image
    """
    def __init__(self):
        self.quantity = []

    def set_quantity(self, quantname, quant):
        """ sets a  2d quantity. the quantity can be anything
        as long as it's in nl by nw
        Parameters
        ----------
        quant : 2d np.ndarray
            the array
        quantname : str
            name of the quantity
        """
        self.quantity.append(quantname)
        setattr(self, quantname, quant)

    def set_laxis(self, laxis):
        self.laxis = laxis

    def set_xy(self, xpnt, ypnt):
        """ xpnt, ypnt in cm
        """
        self.xpnt = xpnt
        self.ypnt = ypnt

    def set_w(self, w):
        self.w = w
        self.nw = len(self.w)

    def get_w(self):
        """
        calculate the wavelength if we know the frequency
        """
        self.w = natconst.cc * 1e4 / self.f

    def get_f(self):
        """
        calculate the frequency if we know the wavelength
        """
        self.f = natconst.cc / self.w * 1e4

    def set_dpc(self, dpc):
        self.dpc = dpc

    def set_restfreq(self, restfreq):
        """ set the rest frequency for a line
        """
        self.restfreq = restfreq

    def get_velocity(self):
        """ get the line of sight velocity for lines
        in cm/s
        """
        if hasattr(self, 'f') is False:
            self.get_f()

        if hasattr(self, 'restfreq') is False:
            raise ValueError('restfreq must be known to get the velocity')

        self.v = natconst.cc * (self.restfreq - self.f) / self.restfreq

    def get_polarization(self, thres=3, fill_value=0):
        """
        a convenient function calculate all the polarization for each wavelength
        """
        # iterate through keys
        keys = ['lpi', 'pi', 'lpa', 'lpol', 'pol', 'cpol']

        for ikey in keys:
            self.set_quantity(ikey, np.zeros(self.I.shape))

        # iterate through wavelength
        for i in range(self.nw):
            out = utils.get_polarization(self.I[...,i], self.Q[...,i], self.U[...,i], self.V[...,i], self.I_rms, self.Q_rms, self.U_rms, self.V_rms, thres=thres, fill_value=fill_value)

            for ikey in keys:
                getattr(self, ikey)[...,i] = out[ikey]

    def write_fits(self, quant, fname):
        """ write to a fits file
        """
        hdu = fits.PrimaryHDU(getattr(self, quant).T)

        # prepare some stuff
        setvals = [
            # keyword, value, comment
            ['NAXIS', 2, ''],

            ['BTYPE', 'Intensity', ''],
            ['BUNIT', 'Jy/beam ', 'Brightness (pixel) unit'],
            ['BSCALE', 1.0, ''],
            ['BZERO', 0.0, ''],

            # write the axis stuff in: naxis, ctype, crval, crpix, cdelt, cunit
            # spatial axis
            ['NAXIS1', len(self.laxis), ''],
            ['CTYPE1', 'OFFSET ', ''],
            ['CRVAL1', self.laxis[0] / natconst.au / self.dpc, ''],
            ['CRPIX1', 1, ''],
            ['CDELT1', (self.laxis[1] - self.laxis[0]) / natconst.au / self.dpc, ''],
            ['CUNIT1', 'arcsec', ''],

            # frequency axis
            ['NAXIS2', len(self.f), ''],
            ['CTYPE2', 'FREQ', ''],
            ['CRVAL2', self.f[0], ''],
            ['CRPIX2', 1, ''],
            ['CDELT2', self.f[1] - self.f[0], ''],
            ['CUNIT2', 'Hz', ''],

            # stokes axis 
#            ['NAXIS3', 1, ''], 
#            ['CTYPE3', 'STOKES', ''],
#            ['CRVAL3', 1, ''], 
#            ['CDELT3', 1, ''], 
#            ['CRPIX3', 1, ''], 
#            ['CUNIT3', ' ', ''], 

            # others
            ['SPECSYS', 'LSRK', ''],

            # not sure what these are
            ['PC1_1', 1.0, ''],
            ['PC2_1', 0.0, ''],
            ['PC3_1', 0.0, ''],

            ['PC1_2', 0.0, ''],
            ['PC2_2', 1.0, ''],
            ['PC3_2', 0.0, ''],

            ['PC1_3', 0.0, ''],
            ['PC2_3', 0.0, ''],
            ['PC3_3', 1.0, ''],
            ]

        if hasattr(self, 'psfarg'):

            # sometimes bmaj depends on the wavelength
            # in which case, we just pick one
            try:
                bmaj = self.psfarg['bmaj'][0] / 3600
                bmin = self.psfarg['bmin'][0] / 3600
                bpa = self.psfarg['bpa'][0]
            except:
                bmaj = self.psfarg['bmaj'] / 3600
                bmin = self.psfarg['bmin'] / 3600
                bpa = self.psfarg['bpa']

            setvals.extend( [
                ['BMAJ', bmaj, ''],
                ['BMIN', bmin, ''],
                ['BPA', bpa, ''],
                ])

        if hasattr(self, 'restfreq'):
            setvals.append(['RESTFRQ', self.restfreq, ''])

        # now fill in the header
        for icombo in setvals:
            hdu.header.set(icombo[0], icombo[1], icombo[2])

        hdul = fits.HDUList([hdu])

        hdul.writeto(fname)

    # ==== plotting ====
    def plot(self, quant='I', iwav=0, ax=None,
        scale_x=1., scale_y=1., **kwargs):
        """ plot the profile as a function laxis
        """
        prof = getattr(self, quant)

        if ax is None:
            ax = plt.gca()

        ax.plot(self.laxis / scale_x, prof[:,iwav] / scale_y, **kwargs)

# ==== 
# image grid 
# ====
class rectangularGrid(object):
    """
    It's a lot easier to separate out the grid info

    Attributes 
    ----------
    x : 1d ndarray
    y : 1d ndarray
    f : 1d ndarray
    w : 1d ndarray
    """
    def __init__(self):
        pass
    def set_xy(self, x, y):
        """
        """
        self.x = x
        self.nx = len(x)

        self.y = y
        self.ny = len(y)

    def set_w(self, w):
        self.w = w
        self.nw = len(self.w)

    def get_w(self):
        """
        calculate the wavelength if we know the frequency
        """
        self.w = natconst.cc * 1e4 / self.f

    def get_f(self):
        """
        calculate the frequency if we know the wavelength
        """
        self.f = natconst.cc * 1e4 / self.w

    def set_dpc(self, dpc):
        """ distance in pc, optional, but necessary for some calculations
        """
        self.dpc = dpc

    def set_restfreq(self, restfreq):
        """ set the rest frequency for a line
        """
        self.restfreq = restfreq

    def get_velocity(self):
        """ get the line of sight velocity for lines
        in cm/s
        """
        if hasattr(self, 'f') is False:
            self.get_freqeuncy()

        if hasattr(self, 'restfreq') is False:
            raise ValueError('restfreq must be known to get the velocity')

        self.v = natconst.cc * (self.restfreq - self.f) / self.restfreq

    def recenter(self, x0, y0):
        """ reset the center
        x0, y0 : float 
            the center in the current coordinate system 
        """
        self.x -= x0
        self.y -= y0


# ====
# intensity image 
# ====
class intensity(object):
    """ 
    the intensity image with some unit. 
    All quantities should be in dimensions of x by y by w
    Attributes 
    ----------
    psfarg : dict, optional
        basic information of the gaussian beam
        bmaj = beam major axis in arcsec
        bmin = beam minor axis in arcsec
        bpa = beam position angle in degrees
    quantity : list
    I : 3d ndarray
    Q, U, V : 3d ndarray, optional

    stokes_unit : string
        the unit of IQUV

    lpi : 3d ndarray, optional
        linear polarized intensity

    lpol : 3d ndarray, optional 
        linear polarization fraction 

    lpa : 3d ndarray, optional
        linear polarization angle

    cpol : 3d ndarray, optional 
        circular polarization fraction 

    pol : 3d ndarray, optional 
        polarization fraction 
    """
    def __init__(self):
        self.quantity = [] # keep track of the physical quantity names

    def set_grid(self, grid):
        self.grid = grid

    def set_quantity(self, quantname, quant):
        """ sets a  3d quantity. the quantity can be anything
        as long as it's in nx by ny by nw
        Parameters
        ----------
        quant : 3d np.ndarray
            the array
        quantname : str
            name of the quantity
        """
        self.quantity.append(quantname)
        setattr(self, quantname, quant)

    def set_stokes_unit(self, stokes_unit):
        """ determine the unit of the stokes value
        """
        if stokes_unit not in ['cgs', 'jyppix', 'jypbeam']:
            raise ValueError('stokes_unit unknown: %s'%stokes_unit)
        self.stokes_unit = stokes_unit

    def set_rms(self, rms, stokes='I'):
        """ 
        set the rms value which is useful for observational data
        assume that the noise level is fixed for each wavelength

        Parameters
        ----------
        rms : float
        """
        key = '%s_rms'%stokes
        setattr(self, key, rms)

    def trim(self, par):
        """ take out a section of the image
        arguments in par
            xlim, ylim, flim
        """
        if 'xlim' in par:
            # understand the direction
            if (self.grid.x[1] - self.grid.x[0]) > 0:
                x_a = np.argmin(abs(min(par['xlim']) - self.grid.x))
                x_b = np.argmin(abs(max(par['xlim']) - self.grid.x)) + 1
            else:
                x_a = np.argmin(abs(max(par['xlim']) - self.grid.x))
                x_b = np.argmin(abs(min(par['xlim']) - self.grid.x)) + 1

            self.grid.x = self.grid.x[x_a:x_b]
            self.grid.nx = len(self.grid.x)
        else:
            x_a = 0
            x_b = self.grid.nx

        if 'ylim' in par:
            # understand the direction
            if self.grid.y[1] - self.grid.y[0] > 0:
                y_a = np.argmin(abs(min(par['ylim']) - self.grid.y))
                y_b = np.argmin(abs(max(par['ylim']) - self.grid.y)) + 1
            else:
                y_a = np.argmin(abs(max(par['ylim']) - self.grid.y))
                y_b = np.argmin(abs(min(par['ylim']) - self.grid.y)) + 1

            # trim
            self.grid.y = self.grid.y[y_a:y_b]
            self.grid.ny = len(self.grid.y)
        else:
            y_a = 0
            y_b = self.grid.nx

    def get_polarization(self, thres=3, fill_value=0):
        """
        a convenient function calculate all the polarization for each wavelength
        """
        # iterate through keys
        keys = ['lpi', 'pi', 'lpa', 'lpol', 'pol', 'cpol']

        for ikey in keys:
            self.set_quantity(ikey, np.zeros(self.I.shape))

        # iterate through wavelength
        for i in range(self.grid.nw):
            out = utils.get_polarization(self.I[...,i], self.Q[...,i], self.U[...,i], self.V[...,i], self.I_rms, self.Q_rms, self.U_rms, self.V_rms, thres=thres, fill_value=fill_value)

            for ikey in key:
                getattr(self, ikey)[...,i] = out[ikey]

    def cut_slit(self, laxis, quant=['I'], width=1,
        trackkw={'x0':0, 'y0':0, 'theta':0}):
        """ instead of interpolating, we use a slit and average the pixels perpendicular to the cut. 
        The slit can only be linear

        Parameters
        ----------
        laxis : 1d ndarray
            the location in the slit direction in cm
        width : int
            The number of pixels. Default is 1
        """
        # check if it's Rectangular grid
        if isinstance(self.grid, rectangularGrid) is False:
            raise ValueError('grid must be a rectangular grid')

        # the angle in the coordinates of the image
        angle = trackkw['theta']

        # unit direction of the slit
        vec =  (np.cos(angle), np.sin(angle))

        # the direction perpendicular to the slit
        per = (np.cos(angle + np.pi/2), np.sin(angle + np.pi/2))

        # the pixel length in the direction perpendicular to the slit
        if (-1 <= np.tan(angle+np.pi/2)) & (np.tan(angle+np.pi/2) <= 1):
            dp = self.grid.dx / np.cos(angle+np.pi/2)
        else:
            dp = self.grid.dy / np.sin(angle+np.pi/2)
        dp = abs(dp)

        # the different centers
        if np.mod(width, 2) == 1:
            offset = np.arange(-(width//2), width//2 + 1)
        else:
            offset = np.arange(- (width//2), width//2) + 0.5
        xcen = trackkw['x0'] + offset * per[0] * dp
        ycen = trackkw['y0'] + offset * per[1] * dp

        # ==== begin interpolation ====
        profs = []
        for iquant in quant:
            # interpolate along different lines
            prof = np.zeros([len(laxis), self.grid.nw, width])
            for i in range(width):
                # the coordinates for this line
                xpnt = xcen[i] + laxis * vec[0]
                ypnt = ycen[i] + laxis * vec[1]

                # convert to pixel coordinates
                x = np.interp(xpnt, self.grid.x, np.arange(self.grid.nx))
                y = np.interp(ypnt, self.grid.y, np.arange(self.grid.ny))

                 # old method: calculate the coordinate grid
#                xi, dum = np.meshgrid(x, z, indexing='ij')
#                yi, dum = np.meshgrid(y, z, indexing='ij')
#                dum, zi = np.meshgrid(np.ones(len(x)), z, indexing='ij')
#                prof[:,:,i] = map_coordinates(getattr(self, iquant), [xi,yi,zi], order=0)

                # iterate through wavelength
                for j in range(self.grid.nw):
                    prof[:,j,i] = map_coordinates(getattr(self, iquant)[...,j], [x,y], order=1)

            profs.append( np.mean(prof, axis=2) )

        # prepare the imageCut object
        cut = imageCut()
        cut.set_laxis(laxis)
        cut.set_xy(trackkw['x0'] + laxis*vec[0], trackkw['y0'] + laxis*vec[1])
        cut.set_w(self.grid.w)
        cut.get_f()
        cut.stokes_unit = self.stokes_unit

        for i, iquant in enumerate(quant):
            cut.set_quantity(iquant, profs[i])

        # some additional attributes
        if hasattr(self.grid, 'dpc'):
            cut.set_dpc(self.grid.dpc)

        if hasattr(self.grid, 'restfreq'):
            cut.set_restfreq(self.grid.restfreq)
            cut.get_velocity()

        for ikey in ['psfarg']:
            setattr(cut, ikey, getattr(self, ikey))

        return cut

# ==== reading the fits file ====
def read_fits(fname, stokes='I'):
    """
    convenient tool to read the image from a fits file for a single quantity
    Note that x should be north and y should be east
    """
    hdul = fits.open(fname)
    hdr = hdul[0].header
    dat = hdul[0].data

    # ==== setup the grid ====
    grid = rectangularGrid()

    # the x-axis is north
    grid.nx = hdr['naxis2']
    grid.dx = hdr['cdelt2'] * 3600 # convert to arcseconds
    grid.x = (np.arange(grid.nx) - (hdr['crpix2']-1) ) * grid.dx
    grid.dec = (np.arange(grid.nx) - (hdr['crpix2']-1)) * hdr['cdelt2'] + hdr['crval2']

    # the y-axis is east
    grid.ny = hdr['naxis1']
    grid.dy = hdr['cdelt1'] * 3600
    grid.y = (np.arange(grid.ny) - (hdr['crpix1']-1)) * grid.dy
    grid.ra = (np.arange(grid.ny) - (hdr['crpix1']-1)) * hdr['cdelt1'] + hdr['crval1']


    # see if there is a frequency axis
    try:
        grid.nw = hdr['naxis3']
        grid.f = (np.arange(grid.nw) - (hdr['crpix3'] - 1) ) * hdr['cdelt3'] + hdr['crval3']
        grid.get_w()

    except:
        grid.nw = 1

    # ==== setup the image ====
    # create the image
    im = intensity()
    im.set_grid(grid)

    # usually, ra is decreasing
    # change the ordering so that it is increasing
    if grid.dy < 0:
        dat = dat[...,::-1]
        im.grid.dy *= -1
        im.grid.y *= -1

    # count the dimensions of the data
    # this determines which 
    dim = dat.shape
    if len(dim) == 2:
        im.set_quantity(stokes, dat[:,:,None])
    elif len(dim) == 3:
        im.set_quantity(stokes, np.moveaxis(dat, 0, -1))
    else:
        raise ValueError('Currently, unacceptable data dimensions for read_fits. It might be better to customize a function to read your fits file.')

    im.set_stokes_unit('jypbeam')

    im.psfarg = {
        'bmaj': hdr['bmaj'] * 3600, 
        'bmin': hdr['bmin'] * 3600, 
        'bpa': hdr['bpa'], 
        }
    hdul.close()

    return im

