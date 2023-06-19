# plot_image.py
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tlx

# ==== reading the images ====
# stokes I
# the name of the fits file
fname = 'Band6_I.fits'

# this line reads the fits file and stores it as a tlx.image.intensity object
im = tlx.image.read_fits(fname, stokes='I')

# in this case, the image only has Stokes I, and we need to collect the other Stokes parameters

# stokes Q
fname = 'Band6_Q.fits'

# we want to use the function to read the image
dum = tlx.image.read_fits(fname, stokes='Q')

# and store it to the im object
im.set_quantity('Q', dum.Q)

# stokes U
fname = 'Band6_U.fits'
dum = tlx.image.read_fits(fname, stokes='U')
im.set_quantity('U', dum.U)

# stokes V
fname = 'Band7_V.fits'
dum = tlx.image.read_fits(fname, stokes='V')
im.set_quantity('V', dum.V)

# sometimes the frequency information is not included in the fits file, 
# in that case we need to manually set the wavelength (or frequency)
im.grid.set_w(np.array([1300]))

# ==== plotting ====
"""
the x and y-axes correspond to Dec (north) and RA (east)
"""
ax = plt.gca()
extent = (im.grid.y[0], im.grid.y[-1], im.grid.x[0], im.grid.x[-1])
pc = ax.imshow(im.I[:,:,0].T, origin='lower', extent=extent)

# usually, we want RA to increase to the left
ax.invert_xaxis()

plt.show()

