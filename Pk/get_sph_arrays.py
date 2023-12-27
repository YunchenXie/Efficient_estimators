import numpy
from scipy.special import sph_harm
import os
from funcs import  get_k_direction, get_real_direction
from nbodykit.lab import *
import time


"""
We notice that for MD-Patchy Mocks, there is a common set of randoms catalog, which means 
We can share a set of Y_lm(r)
"""

st = time.time()

"""
Parameter settings
"""
Nmesh = [512,512,512]
BoxSize = numpy.array([6900,6900,6900]).astype("float64")
poles = [2,4] ## 2,4,6,8...


# Create storage folder
store_path = os.path.join(os.getcwd(),"Ylm_arrays")

if os.path.exists(store_path):
    pass
else:
    os.mkdir(store_path)



k_direction = get_k_direction(Nmesh)
print("k_direction generated")


## Part1 k-space
# for Y_lms and radial binning
for ell in poles:
    for m in range(0,ell+1):
        to_store = sph_harm(m,ell,k_direction[1],k_direction[0])
        store_file = os.path.join(store_path,"Ylm_%s_%s_k.npy"%(ell,m))
        numpy.save(store_file,to_store)
        print("Ylm_%s_%s_k.npy generated"%(ell,m))


## Part2 Config-space
randoms_path = '/public1/home/scg1018/share/Yunchen/desi_y1_blinded/combined_randoms/combined_QSO_NGC_clustering.ran.fits'
randoms = FITSCatalog(randoms_path)


# Select the Correct Redshift Range
ZMIN = 0.8
ZMAX = 2.1

# slice the randoms
valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
randoms = randoms[valid]

cosmo = cosmology.Cosmology(h=0.6736).match(Omega0_m=0.31377) ## planck 2018

## add Cartesian position column
randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

# Add the Completeness Weights
# randoms['WEIGHT'] = 1.0 ## for md_patchy mocks' random file, we already have created this col in the fits file

## Get Box_Length and BoxCenter 
posi =numpy.array(randoms["Position"])
BoxCenter = numpy.array([0,0,0]).astype("float64")
Box_Length = numpy.array([0,0,0]).astype("float64")

for i in range(3):
    BoxCenter[i] = (numpy.max(posi[:,i])+numpy.min(posi[:,i]))/2
    Box_Length[i] = numpy.max(posi[:,i])-numpy.min(posi[:,i])

print("BoxCenter =",BoxCenter)  
print("Box_Length=",Box_Length)  

## Get distance of every mesh grid to the mesh-center in x-space (configuration space)
real_direction = get_real_direction(Nmesh,BoxSize,BoxCenter)
print("real_direction generated")

for ell in poles:
    for m in range(0,ell+1):
        to_store = numpy.conj(sph_harm(m,ell,real_direction[1],real_direction[0]))
        store_file = os.path.join(store_path,"Ylm_%s_%s_x_star.npy"%(ell,m))
        numpy.save(store_file,to_store)
        print("Ylm_%s_%s_x_star.npy generated"%(ell,m))

et = time.time()
print("Total time cost is %s s"%(et-st))

