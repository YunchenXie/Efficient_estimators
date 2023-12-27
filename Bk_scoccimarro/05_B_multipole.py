import numpy
import time
from nbodykit.lab import *
from nbodykit.source.mesh.catalog import CompensateCIC,CompensateTSC,CompensatePCS
import os
import shutil
from funcs import get_k_dist, listdir, space_inversion, get_sub_index, get_sub_config
from pmesh import pm
import gc


"""
PARAMETER SETTING
==================================================================
(All inside "***" block)

evaluate_B4: bools
    Whether or not evaluate B4 

silent: bool
    Whether or not print some additional information which can be very useful in debugging

low_memory_mode: bool
    Whether or not store some binned arrays to disk to save memory.  Make sure that your 
    personal or super computers are equipped with high-speed SSD(e.g you can write or read
    a 2GB ndarray file within 5 seconds) and you have enough frees space to store these 
    temporary files (k_num * 2GB at least). This mode may help some personal computers be 
    able to run a 512-grid measurement with limited chosen triangle configurations

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

mass_assignment: String
    pcs, cic, tsc (https://arxiv.org/abs/astro-ph/0409240v2)

index_range: list
    Batch processing, appoint the indexes of FKP_field or N_field need to be dealt with

FKP_source, N_source,Ylm_source, attrs_source,res_store_path: strings
    Paths to read or store files 

"""


# *************************************************************************************
settings_dict = numpy.load('settings.npy',allow_pickle=True).item()
evaluate_B4 = True
silent = settings_dict["silent"] 
low_memory_mode = False
Nmesh = settings_dict["Nmesh"] 
BoxSize = settings_dict["BoxSize"] .astype("float64")
vol_per_cell = BoxSize.prod()/Nmesh.prod() ## Physical volume of a cell in confif-space

k_mode = 2*numpy.pi/BoxSize
Dk = settings_dict["number_of_kmode"] * k_mode[0]
k_num = settings_dict['k_num']
k_center = (numpy.arange(k_num)+1.5)*Dk 
k_edge = k_center-0.5*Dk 
k_edge = numpy.append(k_edge,k_center[-1]+0.5*Dk)
fixed_k = [[10.5,10.5]] ## fixed k2,k3

mass_assignment = settings_dict['mass_assignment']

## Here we assume that FKP_source, N_source... are stored in current path...
FKP_source = os.path.join(settings_dict['npy_store_path'],"FKP_fields")
N_source = os.path.join(settings_dict['npy_store_path'],"N_fields")
attrs_source = os.path.join(settings_dict['npy_store_path'],"attrs_stores")
Ylm_source = os.path.join(os.getcwd(),"Y_lm_arrays")
res_store_path = os.path.join(os.getcwd(),"Results")
res_collection = settings_dict["res_collection"] ## Collect all results to one dir
# *************************************************************************************



# Define some functions 
# =====================================================================================
"""
Eq 2.7 in https://arxiv.org/abs/1704.02357 and we further apply the space-inversion method.
"""

def get_F_ell(ell):
    for m in range(0,ell+1):
        weighted_field = rfield * numpy.load(os.path.join(Ylm_source,"Ylm_%s_%s_x.npy"%(ell,m)))
        weighted_field = weighted_field.r2c()
        weighted_field.apply(out=Ellipsis,func = cps,kind = "circular")
        weighted_field *= numpy.load(os.path.join(Ylm_source,"Ylm_%s_%s_k_star.npy"%(ell,m)))
        if m != 0:
            F_ell += weighted_field 
            F_ell += numpy.conj(space_inversion(weighted_field)) 
            # Spatial inversion, although this step is not strictly correct in the specific operation 
            # (for example, the point at -N/2 has no inversion counterpart, but we do not need them)
        else:
            F_ell = weighted_field.copy()
        if not silent:
            print("ell=%s,|m|=%s done"%(ell,m))

    return F_ell * (BoxSize.prod() * 4*numpy.pi / (2*ell + 1)) 

## Define a func to deal with binned shotnoise
def get_binned_noise(noise_field):
    binned_noise = []
    for i in range(k_num):
        chosen = numpy.logical_and(dist_arr>=k_edge[i],dist_arr<k_edge[i+1])
        binned_noise.append(numpy.mean(noise_field[chosen]))
    return binned_noise


# Some preparation
# =====================================================================================
## (2*pi) related to FFTs
coef_pi = (2*numpy.pi)**(3+3+3-3) 

## Load VT from disk
VT = numpy.loadtxt("VT_FFT.txt")

"""
Waring: 
If you are using a Mac, hidden files in the path like ".DS_Store" may trouble you 
and trigger a program error
Get the file list
"""
FKP_list,N_list,attrs_list = [],[],[]
listdir(FKP_source, FKP_list, True)
listdir(N_source, N_list, True)
listdir(attrs_source, attrs_list, True)
FKP_list.sort()
N_list.sort()
attrs_list.sort()

# deleta file names contain ".DS_Store"
for X_list in [FKP_list, N_list, attrs_list]:
    for item in X_list: 
        if "Store" in item:
            X_list.remove(item)

        

# Create the result-storage path 
if os.path.exists(res_store_path):
    pass
else:
    os.mkdir(res_store_path)
if not silent:
    print("Result-storage path created")


# get the dist_arr in k-space for radial binning
dist_arr = get_k_dist(Nmesh,k_mode) 
if not silent:
    print("Distance array in k-space has been generated")

"""
Create a ParticleMesh by loading the numpy.ndarray like FKP_field or N_field generated 
by the preprocessing code
"""
my_PM = pm.ParticleMesh(BoxSize=BoxSize, Nmesh=Nmesh,dtype='c16')

## Compensation in k-space:
if mass_assignment == "pcs":
    cps = CompensatePCS
elif mass_assignment == "cic":
    cps = CompensateCIC
elif  mass_assignment == "tsc":
    cps = CompensateTSC

## Full configurations(or you can select some appointed configurations)
sides = numpy.arange(k_num)+1.5
full_config = []
count1 = 0
for i in sides:
    for j in sides:
        for k in sides:
            if i <= j and j <= k and k<=i+j-1.50 and k >= j-i+1.50:
                count1 += 1
                full_config.append([i,j,k])
if not silent:
    print("Totall number of configuration is ", count1)

## get sub_configs and their indexes to chose the corresponding VT
sub_configs, sub_indexes, sub_VTs = [],[],[]
for kk in fixed_k:
    sub_config = get_sub_config(kk[0],kk[1],sides)
    sub_index = get_sub_index(sub_config,full_config)
    for ii in sub_index:
        sub_VTs.append(VT[ii])
    sub_configs.append(sub_config)
    sub_indexes.append(sub_index)
sub_VTs = numpy.array(sub_VTs)
if not silent:
    print("sub_configs =",sub_configs)


# Let's computing the Bis_est
# =====================================================================================
for s in range(0,len(FKP_list)): ## Freedom to adjust according to demand
    st = time.time()

    B_multi = []
    ## Load some attrs like N0 and norm 
    attrs_file = os.path.join(attrs_source,attrs_list[s])
    attrs = numpy.loadtxt(attrs_file)
    I33, S_00, norm, BoxCenter = attrs[1], attrs[2],attrs[3],attrs[4:]

    ## Create the realfield Object(FKP_field), FFT and Compensate it 
    fkp_name = os.path.join(FKP_source, FKP_list[s])
    rfield = my_PM.create(type="real", value=numpy.load(fkp_name))
    cfield = rfield.r2c()
    cfield.apply(out = Ellipsis, func = cps, kind = "circular")

    ## Load N_field from disk, FFT and Compensate it 
    N_name = os.path.join(N_source, N_list[s])
    Nfield = my_PM.create(type="real", value=numpy.load(N_name))
    K_Nfield = Nfield.r2c() * norm
    K_Nfield.apply(out = Ellipsis, func = cps, kind = "circular")

    """
    Get F0, then FFT binned F0s to x-space
    Create a folder to store the binned FFTs of F0 in low_memory_mode
    We'd better delete them after all multipoles achieved
    """
    F0 = cfield
    if not low_memory_mode:
        binned_x_F0 = []
        for i in range(k_num):
            chosen = numpy.logical_and(dist_arr>=k_edge[i],dist_arr<k_edge[i+1])
            binned_F0 = F0.copy()
            binned_F0[~chosen] = 0
            binned_x_F0.append(numpy.array(binned_F0.c2r()))
    else:
        F0_path = './F0/'
        if os.path.exists(os.path.join(os.getcwd(),F0_path)):
            pass
        else:
            os.mkdir(os.path.join(os.getcwd(),F0_path))

        for i in range(k_num):
            chosen = numpy.logical_and(dist_arr>=k_edge[i],dist_arr<k_edge[i+1])
            binned_F0 = F0.copy()
            binned_F0[~chosen] = 0
            to_store = numpy.array(binned_F0.c2r())
            numpy.save(F0_path + "%s_bin.npy"%i, to_store)

    if not silent:
        print("Time spent on getting binned F0-arrays is %s s"%(time.time()-st))


    ## Part 1: Get Monopole
    # ------------------------------------------------------------------------------
    """
    Eq.53,54 in https://arxiv.org/abs/1506.02729
    If you have already appointed some chosen configs (e.g. with k1 and k2 fixed) in low_memory_mode
    then we can load binned F0 fields k1 and k2 only from disk only time as a temp-field
    which may help you save a lot of time spent on file loading and extend the life 
    of your SSD
    """
    B_mono = []
    if not low_memory_mode:
        for j in range(len(fixed_k)):
            for xx in sub_configs[j]:
                ind_0,ind_1,ind_2 = int(xx[0]-1.5),int(xx[1]-1.5),int(xx[2]-1.5)
                B_mono.append(numpy.sum(binned_x_F0[ind_0]\
                                        * binned_x_F0[ind_1]*binned_x_F0[ind_2]))
    else:
        for j in range(len(fixed_k)):
            ind_0,ind_1= int(fixed_k[j][0]-1.5),int(fixed_k[j][1]-1.5)
            temp_field = numpy.load(F0_path + "%s_bin.npy"%ind_0) \
                * numpy.load(F0_path + "%s_bin.npy"%ind_1)
            for xx in sub_configs[j]:
                ind_2 = int(xx[2]-1.5)
                B_mono.append(numpy.sum(temp_field \
                                        * numpy.load(F0_path + "%s_bin.npy"%(ind_2))))

    B_mono = coef_pi * vol_per_cell/(I33*sub_VTs) * numpy.real(numpy.array(B_mono))
    numpy.savetxt(os.path.join(res_store_path,"B_mono.txt"),B_mono)
    B_multi.append(B_mono)
        

    ## Shotnoise of B_monopole
    noise_field_B0 = (F0 * BoxSize.prod()) * numpy.conj(K_Nfield) 
    binned_noise_B0 = get_binned_noise(noise_field_B0)

    B0_noise = []
    for j in range(len(fixed_k)):
        for xx in sub_configs[j]:
            ind_0,ind_1,ind_2 = int(xx[0]-1.5),int(xx[1]-1.5),int(xx[2]-1.5)
            B0_noise.append(binned_noise_B0[ind_0]+\
                            binned_noise_B0[ind_1]+binned_noise_B0[ind_2])

    B0_noise = (numpy.real(numpy.array(B0_noise))-2*S_00)/I33 
    numpy.savetxt(os.path.join(res_store_path,"B0_noise.txt"),B0_noise)
    B_multi.append(B0_noise)

    del noise_field_B0
    gc.collect()
    
    if not silent:
        print("The time spent to complete the calculation of B0 is %s s"%(time.time()-st))


    ## Part 2: Get quadrupole
    # ------------------------------------------------------------------------------
    F2 = get_F_ell(2)
    print("xxx"*50)

    B_quad = []
    if not low_memory_mode:
        for j in range(len(fixed_k)):
            ind_0 = int(fixed_k[j][0]-1.5)
            chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
            binned_F2 = F2.copy()
            binned_F2[~chosen] = 0
            binned_F2 = numpy.array(binned_F2.c2r())
            for xx in sub_configs[j]:
                ind_1,ind_2 = int(xx[1]-1.5),int(xx[2]-1.5)
                B_quad.append(numpy.sum(binned_F2\
                                        * binned_x_F0[ind_1]*binned_x_F0[ind_2]))
    else:
        for j in range(len(fixed_k)):
            ind_0,ind_1= int(fixed_k[j][0]-1.5),int(fixed_k[j][1]-1.5)
            chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
            binned_F2 = F2.copy()
            binned_F2 = binned_F2[~chosen]
            binned_F2 = numpy.array(binned_F2.c2r()) 
            temp_field = binned_F2 * numpy.load(F0_path + "%s_bin.npy"%ind_1)
            for xx in sub_configs[j]:
                ind_2 = int(xx[2]-1.5)
                B_quad.append(numpy.sum(temp_field \
                                        * numpy.load(F0_path + "%s_bin.npy"%(ind_2))))

    B_quad = 5 * coef_pi * vol_per_cell/(I33*sub_VTs) * numpy.real(numpy.array(B_quad))
    numpy.savetxt(os.path.join(res_store_path,"B_quad.txt"),B_quad)
    B_multi.append(B_quad)


    ##  shotnoise of quadrupole
    B2_noise = []
    noise_field_B2 = (F2 * BoxSize.prod()) * numpy.conj(K_Nfield)
    for j in range(len(fixed_k)):
        ind_0 = int(fixed_k[j][0]-1.5)
        chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
        N2_q1 = 5 * numpy.mean(noise_field_B2[chosen])/I33
        for xx in sub_configs[j]:
            B2_noise.append (N2_q1)
    B2_noise = numpy.real(numpy.array(B2_noise))
    numpy.savetxt(os.path.join(res_store_path,"B2_noise.txt"),B2_noise)
    B_multi.append(B2_noise)

    del chosen, noise_field_B2,F2
    gc.collect()


    ## Part 3: Get hexadecapole
    # ------------------------------------------------------------------------------
    if evaluate_B4:
        F4 = get_F_ell(4)
        B_hexa = []
        if not low_memory_mode:
            for j in range(len(fixed_k)):
                ind_0 = int(fixed_k[j][0]-1.5)
                chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
                binned_F4 = F4.copy()
                binned_F4[~chosen] = 0
                binned_F4 = numpy.array(binned_F4.c2r()) 
                for xx in sub_configs[j]:
                    ind_1,ind_2 = int(xx[1]-1.5),int(xx[2]-1.5)
                    B_hexa.append(numpy.sum(binned_F4\
                                            * binned_x_F0[ind_1]*binned_x_F0[ind_2]))
        else:
            for j in range(len(fixed_k)):
                ind_0,ind_1= int(fixed_k[j][0]-1.5),int(fixed_k[j][1]-1.5)
                chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
                binned_F4 = F4.copy()
                binned_F4 = binned_F4[~chosen]
                binned_F4 = numpy.array(binned_F4.c2r()) 
                temp_field = binned_F4 * numpy.load(F0_path + "%s_bin.npy"%ind_1)
                for xx in sub_configs[j]:
                    ind_2 = int(xx[2]-1.5)
                    B_hexa.append(numpy.sum(temp_field \
                                            * numpy.load(F0_path + "%s_bin.npy"%(ind_2))))

        B_hexa = 9 * coef_pi * vol_per_cell/(I33*sub_VTs) * numpy.real(numpy.array(B_hexa))
        numpy.savetxt(os.path.join(res_store_path,"B_hexa.txt"),B_hexa)
        B_multi.append(B_hexa)

        ##  shotnoise of hexadecapole
        B4_noise = []
        noise_field_B4 = (F4 * BoxSize.prod()) * numpy.conj(K_Nfield)
        for j in range(len(fixed_k)):
            ind_0 = int(fixed_k[j][0]-1.5)
            chosen = numpy.logical_and(dist_arr>=k_edge[ind_0],dist_arr<k_edge[ind_0+1])
            N4_q1 = 9 * numpy.mean(noise_field_B4[chosen])/I33
            for xx in sub_configs[j]:
                B4_noise.append (N4_q1)
        B4_noise = numpy.real(numpy.array(B4_noise))
        numpy.savetxt(os.path.join(res_store_path,"B4_noise.txt"),B4_noise)
        B_multi.append(B4_noise)

        del chosen, noise_field_B4, rfield, F4
        gc.collect()


    if low_memory_mode:
        shutil.rmtree(os.path.join(os.getcwd(),F0_path),True)
        if not silent:
            print("F0 fields have already been deleted from the disk")

    """
    Let's store the Results
    """
    Res = numpy.array(B_multi)
    f_name = FKP_list[s]
    res_name = "B_multi"+f_name+".txt"
    numpy.savetxt(os.path.join(res_store_path,res_name),Res)
    numpy.savetxt(os.path.join(res_collection,res_name),Res)