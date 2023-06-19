# plot_cut.py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import toolbox as tlx
rad = np.pi / 180

def plot1(cut, quant=['lpi'], mode='major',
    figsize=(12,7), xlim=None, pdfname=None):
    """
    plot quantities with I as a comparison
    """
    nrow, ncol = len(quant), 1
    fig, axgrid = plt.subplots(nrow,ncol,sharex='col', sharey='row',
        squeeze=False, figsize=figsize)
    axes = axgrid.flatten()

    # get the mmaxis for the models
    if mode == 'major':
        xlabel = 'Major axis offset'
    elif mode == 'minor':
        xlabel = 'Minor axis offset'
    else:
        raise ValueError('mode unknown')

    pltset = {
        # scale, quant name, ylim
        'I':[1e-3, r'Stokes $I$'+'\n'+'(mJy beam$^{-1}$)', (0, None)],
        'Q':[1e-3, r'Stokes $Q$', (0, None)],
        'U':[1e-3, r'Stokes $U$', (0, None)],
        'V':[1e-3, r'Stokes $V$', (0, None)],
        'lpi':[1e-3, 'Polarized Intensity'+'\n'+r'(mJy beam$^{-1}$)', (0,None)],
        'lpol':[1e-2, 'Polarization Percent' + '\n' + '(%)', (0,5)],
        'q':[1e-2, 'Q / I [%]', (-5,5)],
        'u':[1e-2, 'U / I [%]', (-5,5)],
        }

    for j in range(len(quant)):
        ax = axes[j]

        # read the settings
        scale = pltset[quant[j]][0]
        quant_name = pltset[quant[j]][1]
        ylim = pltset[quant[j]][2]

        # observations
        pltx = cut.laxis
        plty = getattr(cut, quant[j])[:,0] / scale

        # take out the zeros
        if quant[j] in ['lpi', 'lpol']:
            reg = plty == 0
            pltx[reg] = np.nan

        if quant[j] in ['I', 'lpi', 'q']:
            yerr = getattr(cut, '%s_rms'%(quant[j])) / scale
            ax.plot(pltx, plty, label='observed', color='k')

            ax.fill_between(pltx, plty-yerr, plty+yerr, color='grey', alpha=0.3)
        elif quant[j] in ['lpol']:
            # some error quantities depend on wavelength
            yerr = getattr(cut, '%s_rms'%(quant[j]))[...,0] / scale
            ax.plot(pltx, plty, label='observed', color='k')
            ax.fill_between(pltx, plty-yerr, plty+yerr, color='grey', alpha=0.3)
        else:
            ax.plot(pltx, plty, label='observed', color='k')

        # ylim 
        ax.set_ylim(ylim)

        ax.set_ylabel(quant_name)

        ax.set_xlim(xlim)

        # create a secondary axis for Stokes I
        ax2 = ax.twinx()
        scale = pltset['I'][0]
        quant_name = pltset['I'][1]
        ylim = pltset['I'][2]
        pltx = cut.laxis
        plty = getattr(cut, 'I')[:,0] / scale
        yerr = cut.I_rms / scale
        ax2.plot(pltx, plty, color='C0')
        ax2.fill_between(pltx, plty-yerr, plty+yerr, color='C0', alpha=0.3)

        ax2.set_xlim(xlim)
        ax2.set_ylabel(quant_name)
        ax2.set_ylim(ylim)

    # label the quantities
    for i in range(nrow):
        ax = axes[i]
        ax.axvline(x=0, color='k', linestyle='--')

    axes[-1].set_xlabel(xlabel + ' ["]')

    fig.subplots_adjust(hspace=0.1,
        left=0.10,
        bottom=0.1,
        right=0.90,
        top=0.98)

    if pdfname is not None:
        fig.savefig(pdfname)

    plt.show()


# ==== reading the images ====
# stokes I
fname = 'Band6_I.fits'
im = tlx.image.read_fits(fname, stokes='I')
im.I_rms = 72e-6

# stokes Q
fname = 'Band6_Q.fits'
dum = tlx.image.read_fits(fname, stokes='Q')
im.set_quantity('Q', dum.Q)
im.Q_rms = 15e-6

# stokes U
fname = 'Band6_U.fits'
dum = tlx.image.read_fits(fname, stokes='U')
im.set_quantity('U', dum.U)
im.U_rms = 15e-6

# stokes V
fname = 'Band6_V.fits'
dum = tlx.image.read_fits(fname, stokes='V')
im.set_quantity('V', dum.V)
im.V_rms = 14e-6

# sometimes the frequency information is not included in the fits file, 
# in that case we need to manually set the wavelength (or frequency)
im.grid.set_w(np.array([1300]))

# ==== cutting the image ====
# position angle of the disk
pa = 138.02 * rad

# arguments for the cut
trackkw = {'x0':0, 'y0':0, 'theta':pa}

# the 1d length we want for the cut
laxis = np.arange(-1.5, 1.5, abs(im.grid.dx))

# width of the slit for averaging 
width = 5

# implement the cut
cut = im.cut_slit(laxis, quant=['I','Q','U','V'], trackkw=trackkw, width=width)

# also transfer the noise information to the cut object
for ikey in ['I_rms', 'Q_rms', 'U_rms', 'V_rms']:
    setattr(cut, ikey, getattr(im, ikey))

# we happen to need the lpi rms for easier plotting
cut.lpi_rms = tlx.utils.calc_lpi_rms(cut.Q_rms, cut.U_rms)

# calculate the polarization 
cut.get_polarization(thres=3, fill_value=0)

# also calculate the uncertainty for lpol
cut.lpol_rms = cut.lpol * np.sqrt( (cut.lpi_rms/cut.lpi)**2 + (cut.I_rms/cut.I)**2)

# ==== plotting ====
plot1(cut, quant=['lpi', 'lpol'], mode='major')

