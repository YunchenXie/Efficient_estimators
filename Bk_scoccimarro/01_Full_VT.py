import numpy
import time
from nbodykit.lab import *
import os
from funcs import get_k_dist
from pmesh import pm

"""
PARAMETER SETTING
==================================================================
(All inside "***" block)

silent: bool
    Whether or not print some additional information which can be very useful in debugging

store_path: string
    Path to store the results. Default is local path

Nmesh: 1d array 
    Grid size. Three equal numbers are requested
    
BoxSize: 1d array
    Box size. Three equal numbers are requested

k_mode: 1d array 
    Length of the fundamental boxes in k-space
    $k_f = 2\pi/L$

Dk: float
    Width of the k-bin. Here we set Dk as integral multiple of k_mode

k_num: int
    Number of k_bins

k_center: 1d array
    Geometric centers of k_bins which can be used to calculate V_T analytically.
    Note that there is a small difference between geometric k-center and averaged k-center
    Here our default first k_center is 1.5*Dk(same as Hector),which means we 
    discard the first Dk-width kbin from the origin. 
    Other people(Changhoon or Roman) prefer to set k_center as integral multiple, please 
    modify relative parts in the code if you accept this convention.

k_edge: 1d array
    Edge of kbins. 
    k_edge[i] = k_center[i] - Dk/2, k_edge[i+1] = k_center[i] + Dk/2

dist_arr: 3d array
    Which denotes the distance from original points to each grid in k-space

"""

# *************************************************************************************
settings_dict = numpy.load('settings.npy',allow_pickle=True).item()

silent = settings_dict["silent"] 
store_path = os.getcwd()
Nmesh = settings_dict["Nmesh"] 
BoxSize = settings_dict["BoxSize"].astype("float64")
vol_per_cell = BoxSize.prod() / Nmesh.prod()
k_mode = 2 * numpy.pi / BoxSize
Dk = settings_dict["number_of_kmode"] * k_mode[0]
k_num = settings_dict["k_num"]
k_center = (numpy.arange(k_num) + 1.5) * Dk
k_edge = k_center - 0.5 * Dk
k_edge = numpy.append(k_edge, k_center[-1] + 0.5 * Dk)

# *************************************************************************************

## to store some information which may help get some insight of the running state of the program
count_path = os.path.join(os.getcwd(),"count")
if os.path.exists(count_path):
    pass
else:
    os.mkdir(count_path)

dist_arr = get_k_dist(Nmesh, k_mode)

## Create an ndarray-liked pmesh-object which can be "FFT"ed more 
## quickly than numpy.fft.fftn()
my_PM = pm.ParticleMesh(BoxSize=BoxSize, Nmesh=Nmesh, dtype='c16')
Ones_field = my_PM.create(type="complex", value=numpy.ones(Nmesh))


## Bin the ones-field and FFT each bin to x-space
def get_binned_ones(i):
    chosen = numpy.logical_and(dist_arr >= k_edge[i], dist_arr < k_edge[i + 1])
    binned_ones = Ones_field.copy()
    binned_ones[~chosen] = 0
    return numpy.array(binned_ones.r2c_vjp())


binned_ones = []
for i in range(len(k_center)):
    binned_ones.append(get_binned_ones(i))

## Addtional coefficient related to FFTs.

"""
Note that if you want to use discrete FFT algorithms to perform 
FTs of Continuous functions, this coefficient is necessary 
# coeff = (2*numpy.pi)**6 * (Nmesh.prod()/BoxSize.prod())**2
"""

coeff = (2 * numpy.pi) ** 6 / vol_per_cell ** 2

## Get full configurations of triangles in k-space
sides = numpy.arange(k_num) + 1.5  # in k_mode unit
config = []
count1 = 0
for i in sides:
    for j in sides:
        for k in sides:
            if i <= j and j <= k and k <= i + j - 1.50 and k >= j - i + 1.50:
                count1 += 1
                config.append([i, j, k])
if not silent:
    print("Number of configuration is ", count1)

VT_FFT = []
count = 0
for xx in config:
    ind_0, ind_1, ind_2 = int(xx[0] - 1.5), int(xx[1] - 1.5), int(xx[2] - 1.5)
    # print(ind_0,ind_1,ind_2)
    VT_FFT.append(numpy.sum(binned_ones[ind_0] * binned_ones[ind_1] * binned_ones[ind_2]))

    # if you want...only for checking the calculating progress
    # Nothing really matters...
    if not silent:
        count += 1
        if count % 100 == 0:
            count_name = os.path.join(count_path,"VT_count%s.txt"%count)
            numpy.savetxt(count_name, k_center)
            
VT_FFT = coeff * numpy.real(numpy.array([VT_FFT]))
numpy.savetxt(os.path.join(store_path, "VT_FFT.txt"), VT_FFT)

## We can also easily get VT analytically: $ V_T^{ANA} = 8\pi^2k_1k_2k_3\Delta k^3 $
VT_ANA = []
for xx in config:
    VT_ANA.append(8 * numpy.pi ** 2 * xx[0] * xx[1] * xx[2] * Dk ** 6)
numpy.savetxt(os.path.join(store_path, "VT_ANA.txt"), VT_ANA)
