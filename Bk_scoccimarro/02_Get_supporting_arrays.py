"""
Why this code necessary?
=====================================================================================
We may notice that for all MD-Patchy Mocks, they share only one randoms
catalog, which means we can pre-generate some Y_lm(x) or Y_lm(k) fields
and store them to the disk, this can help us save a lot of time!
For a 512^3 & complex128 ndarray, the file size is 2GB. Modern(Mid-2023) 
super computers and personal computers are commonly equipped with high-speed SSD,
which means we can store and load such ndarrays within 4s, which is much faster than 
generate Y_lm arrays at real-time in Main bispectrum/power spcetrum code.
"""

import numpy
from scipy.special import sph_harm
import os
from funcs import get_k_direction, get_real_direction
from nbodykit.lab import *
import time

# PARAMETER SETTING (All inside "***" block)
# *************************************************************************************
settings_dict = numpy.load('settings.npy',allow_pickle=True).item()

silent = settings_dict["silent"] 
store_path = os.path.join(os.getcwd(), "Y_lm_arrays")  # Specifies the file storage path
Nmesh = settings_dict["Nmesh"] 
BoxSize = settings_dict["BoxSize"] .astype("float64")
poles = [2, 4]  # quadrupole, hexadecapole ... even higher orders
# Fiducial cosmology
cosmo = cosmology.Cosmology(h=settings_dict["cosmo_h"]).match(Omega0_m=settings_dict["cosmo_Omega0"]) 

z_range = settings_dict["z_range"] 
randoms_path = settings_dict["random_catalog_file"] 

# *************************************************************************************


st = time.time()
if not silent:
    print("Start to generate supporting Y_lm arrays")

# Create the storage path 
if os.path.exists(store_path):
    pass
else:
    os.mkdir(store_path)
if not silent:
    print("Storage path created")

# =====================================================================================
## Part1 k-space

"""
Since we already have the space-inversion method, only nonnegative-m terms are needed
"""

k_direction = get_k_direction(Nmesh)
if not silent:
    print("k_direction generated")

for ell in poles:
    for m in range(0, ell + 1):
        to_store = numpy.conj(sph_harm(m, ell, k_direction[1], k_direction[0]))
        store_file = os.path.join(store_path, "Ylm_%s_%s_k_star.npy" % (ell, m))
        numpy.save(store_file, to_store)
        if not silent:
            print("Ylm_%s_%s_k_star.npy generated" % (ell, m))


# =====================================================================================
## Part2 Config-space

"""
We need the randoms catalog to determine BoxCenter
Accuatally this is not necessary, we can load from the stored-file from 
other sub-codes... Anyway, since we only perform this process for one time
Nothing really matters
"""

randoms = FITSCatalog(randoms_path)

# Select the Correct Redshift Range
ZMIN = z_range[0]
ZMAX = z_range[1]

# Slice the randoms
valid = (randoms['Z'] > ZMIN) & (randoms['Z'] < ZMAX)
randoms = randoms[valid]

## Add Cartesian position column
randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

## get the BoxCenter
posi = numpy.array(randoms["Position"])
if not silent:
    print("posi.shape =", posi.shape)
BoxCenter = numpy.array([0, 0, 0]).astype("float64")
for i in range(3):
    BoxCenter[i] = (numpy.max(posi[:, i]) + numpy.min(posi[:, i])) / 2
if not silent:
    print("BoxCenter =", BoxCenter)

real_direction = get_real_direction(Nmesh, BoxSize, BoxCenter)
if not silent:
    print("real_direction generated")

for ell in poles:
    for m in range(0, ell + 1):
        to_store = sph_harm(m, ell, real_direction[1], real_direction[0])
        store_file = os.path.join(store_path, "Ylm_%s_%s_x.npy" % (ell, m))
        numpy.save(store_file, to_store)
        if not silent:
            print("Ylm_%s_%s_x.npy generated" % (ell, m))

et = time.time()
if not silent:
    print("Total time cost is %s s" % (et - st))
