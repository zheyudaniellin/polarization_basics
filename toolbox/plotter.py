"""
plotter.py
This is a collection of functions to plot images

"""
def add_colormap(ax, im, quant, north='right', axis_unit='au', vmax=None):
    """
    add a colormap
    """
    if quant == 'I':
        scale_map = 1e-3
        im2d = im.I[:,:,0] / scale_map
        cmap = plt.cm.gist_heat
        norm = vsl.ImageNormalize(im2d, stretch=vsl.LinearStretch(),
#            interval=vsl.MinMaxInterval(), 
            interval=vsl.ManualInterval(vmin=vmin,vmax=vmax))
    elif quant == 'Q':
        scale_map = 1e-3
        im2d = im.Q[:,:,0] / scale_map
        cmap = plt.cm.RdBu_r

        if vmax is None:
            vmax = np.nanmax(im2d)

        vmin = - vmax
        norm = vsl.ImageNormalize(im2d, stretch=vsl.LinearStretch(),
#            interval=vsl.MinMaxInterval(), 
            interval=vsl.ManualInterval(vmin=vmin,vmax=vmax))

    elif quant == 'U':
        scale_map = 1e-3
        im2d = im.U[:,:,0] / scale_map
        cmap = plt.cm.RdBu_r

        if vmax is None:
            vmax = np.nanmax(im2d)

        vmin = -vmax
        norm = vsl.ImageNormalize(im2d, stretch=vsl.LinearStretch(),
#            interval=vsl.MinMaxInterval(), 
            interval=vsl.ManualInterval(vmin=vmin,vmax=vmax))
    elif quant == 'lpi':
        scale_map = 1e-3
        im2d = im.lpi[:,:,0] / scale_map
        cmap = plt.cm.nipy_spectral

        vmin = 0

        norm = vsl.ImageNormalize(im2d, stretch=vsl.LinearStretch(),
#            interval=vsl.MinMaxInterval(), 
            interval=vsl.ManualInterval(vmin=vmin,vmax=vmax))

    elif quant == 'lpol':
        scale_map = 1
        im2d = im.lpol[:,:,0] / scale_map
        cmap = plt.cm.nipy_spectral
        vmin = 0
        norm = vsl.ImageNormalize(im2d, stretch=vsl.LinearStretch(),
#            interval=vsl.MinMaxInterval(), 
            interval=vsl.ManualInterval(vmin=vmin,vmax=vmax))
    else:
        raise ValueError('quant unknown')

    # determine north 
    if north == 'right':
        pltx = im.grid.x * 1
        plty = im.grid.y * 1
    elif north == 'up':
        pltx = im.grid.y * 1
        plty = im.grid.x * 1
        im2d = im2d.T
    else:
        raise ValueError('north unknown')

    # axis unit
    if axis_unit == 'cm':
        pltx *= im.grid.dpc * au
        plty *= im.grid.dpc * au
    elif axis_unit == 'au':
        pltx *= im.grid.dpc
        plty *= im.grid.dpc
    elif axis_unit == 'arcsec':
        # already in arcsec
    else:
        raise ValueError('axis_unit unknown')

    extent = (pltx[0], pltx[-1], plty[0], plty[-1])

    # colormap
    pc = ax.imshow(im2d.T, origin='lower', extent=extent, cmap=cmap,
        norm=norm)

    return pc

def add_beam(ax, im, axis_unit='arcsec', north='up', beamxy=None, facecolor='w'):
    """
    plot a beam 
    bmaj and bmin in im.psfarg is in arcseconds by default

    Parameters
    ----------
    beamxy : tuple
        location of the center of the beam 
    axis_unit : string
        unit the beam should be and also for beamxy
    north : str
        direction of north. 'right' or 'up'
    """
    
    # determine size of ellipse to overplot
    # conversion factor from arcsec 
    if axis_unit == 'cm':
        fac = im.grid.dpc * natconst.au
    elif axis_unit == 'au':
        fac = im.grid.dpc
    elif axis_unit == 'arcsec':
        fac = 1.
    else:
        raise ValueError('axis_unit unknown: %s'%axis_unit)

    # fetch the bmaj, bmin and bpa
    try:
        bmaj = im.psfarg['bmaj'][iwav]
        bmin = im.psfarg['bmin'][iwav]
        bpa = im.psfarg['bpa'][iwav]
    except TypeError:
        bmaj = im.psfarg['bmaj']
        bmin = im.psfarg['bmin']
        bpa = im.psfarg['bpa']

    # determine the height and width of the ellipse
    if north == 'right':
        ewidth = bmaj
        eheight = bmin
        ang = bpa
    elif north == 'up':
        ewidth = bmin
        eheight = bmaj
        ang = - bpa
    else:
        raise ValueError('north unknown')

    # center of the beam
    if beamxy is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ex = xmin + 0.8 * (xmax - xmin)
        ey = ymin + 0.2 * (ymax - ymin)
    else:
        ex = beamxy[0]
        ey = beamxy[1]

    # set up the ellipse
    ells = Ellipse( xy=(ex,ey),
        width=ewidth*fac, height=eheight*fac, angle=ang)

    ells.set_facecolor(facecolor)
    ells.set_fill(True)
    ax.add_patch(ells)
