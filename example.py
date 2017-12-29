# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pacril

# First we get some of the data to work with which is available in the package,
# specifically the 2'C-2'2' locomotive NSB type 30 which was common in the
# Norwegian  railway network.
locomotives = pacril.data.locomotives()
loc = locomotives['1900-1930']["2'C-2'2'"]

# We then find the load vector for the locomotive.
f = pacril.get_loadvector(loc['p'], loc['xp'])
xf = pacril.get_coordinate_vector(f)

# We also load a influence line available in pacril, the strain influence line
# from a stringer on Hell railway bridge.
l = pacril.data.influencelines()['hell']['stringer']
xl = pacril.get_coordinate_vector(l)

# We find the response of the NSB type 30 locomotive running across the
# simply supported beam.
z = np.convolve(l, f)
xz = pacril.get_coordinate_vector(z)

# We can retrieve the influence line from the response by the functionality
# in the pacril package aswell.
l_lstsq = pacril.find_influenceline_lstsq(z, f)
l_fd = pacril.find_influenceline_fourier(z, f)


# Let us present the data
fig = plt.figure(dpi=300)
axf = plt.subplot2grid((2, 2), (0, 0))
axf.plot(xf, f)
axf.set(ylim = (-1, 18), xlim=(-1, 20),
       title="Load vector NSB {0:s}".format(loc['nsb_class']),
       ylabel='Axle load [t]', xlabel='Axle position [m]')

axl = plt.subplot2grid((2, 2), (0, 1))
axl.plot(xl, l, label='True', lw=3., alpha=.3)
axl.plot(xl, l_lstsq, label='LSTSQ')
axl.plot(xl, l_fd, label='FD')
axl.set(ylabel='Influence line ordinate', xlabel='Load position [m]',
        title="True and estimated influence lines")
axl.legend()

axz = plt.subplot2grid((2, 2), (1, 0), colspan=2)
axz.plot(xz, z)
axz.set(xlabel='Load shift', ylabel='Response',
        title='Response from load passing influence line')
fig.tight_layout()
plt.show(block=True)