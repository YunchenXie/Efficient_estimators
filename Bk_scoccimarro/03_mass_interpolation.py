"""
WARNING!!! 
Very important!
=======================================================================================
If you want to use MPI to accelerate the catalog-to-mesh 
processing, you'd better set mpi size as a prime number e.g. 2,3,5,7...
or you may get the FKP_field or N_field with wrong shape because 
numpy.concatenate() can only concatenate the received sub-fields 
along one dimension
"""

import numpy
import time
from nbodykit.lab import *
import os
import gc
from nbodykit import CurrentMPIComm
from funcs import get_Ncatalog
from scipy.special import sph_harm

comm = CurrentMPIComm.get()
size = comm.Get_size()
print("comm.rank=", comm.rank)

# PARAMETER SETTING (All inside "***" block)
# *************************************************************************************
settings_dict = numpy.load('settings.npy',allow_pickle=True).item()

silent = settings_dict["silent"] 
Nmesh = settings_dict["Nmesh"] 
BoxSize = settings_dict["BoxSize"] .astype("float64")

"""
We may need to deal with the situation that we 
have many mock data catalogs with one random catalog!
"""
## Get the galaxy-fits files need to be dealt with
data_file_list = []
file_source = settings_dict["data_fits_path"]

## Read the random file
randoms_file = settings_dict["random_catalog_file"] 

## Path to store the FKP_field, N_field and attrs
store_path = settings_dict['npy_store_path']

## Redshift selection and fiducial cosmology
z_range = settings_dict["z_range"] 
cosmo = cosmology.Cosmology(h=settings_dict["cosmo_h"]).match(Omega0_m=settings_dict["cosmo_Omega0"]) 

## Interpolation schemes and wether interlaced
mass_assignment = settings_dict['mass_assignment']  ## cic, tsc or pcs
interlaced = settings_dict["interlaced"]

## Batch processing, appoint the indexes of galaxy-fits files need to be dealt with
index_range = settings_dict["index_range"]


# *************************************************************************************
## create some dir to store fields and attrs
if comm.rank == 0:
    print("I am Groot.")
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    for item in ["attrs_stores","FKP_fields","N_fields"]:
        if not os.path.exists(os.path.join(store_path,item)):
            os.mkdir(os.path.join(store_path,item))


def listdir(path, list_name, iscontent):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            _isC = False
            if os.path.basename(file_path) == "TF":
                _isC = True
            listdir(file_path, list_name, _isC)
        else:
            if iscontent:
                list_name.append(file)


"""
Warning:
If you are using a Mac, hidden files in the dir like ".DS_Store" may trouble you 
if they are added to the "data_file_list" which will trigger a program error
"""

listdir(file_source, data_file_list, True)
data_file_list.sort()

for item in data_file_list:
    if "Store" in item:
        data_file_list.remove(item)
        break

## Wirte the galaxy catalog file names into a txt file for checking...
if not silent:
    if comm.rank == 0:
        F = open(r'data_file_list.txt', 'w')
        for i in data_file_list:
            F.write(i + '\n')
        F.close()

    # =====================================================================================
## Generate overdensity fields from data/randoms catalogs
for i in range(index_range[0], index_range[1]):
    start_time = time.time()

    # Initialize the FITS catalog objects for data and randoms
    data = FITSCatalog(os.path.join(file_source, data_file_list[i]))
    randoms = FITSCatalog(randoms_file)

    # Select the Correct Redshift Range
    ZMIN = z_range[0]
    ZMAX = z_range[1]

    # slice the randoms
    valid = (randoms['Z'] > ZMIN) & (randoms['Z'] < ZMAX)
    randoms = randoms[valid]

    # slice the data
    valid = (data['Z'] > ZMIN) & (data['Z'] < ZMAX)
    data = data[valid]

    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

    data["NZ"] = data["NX"]
    randoms["NZ"] = randoms["NX"]

    # Add the Completeness Weights
    # (it depends on your catalogs, for some of them ,these weights are provided already)
    # randoms['WEIGHT'] = 1.0
    # data['WEIGHT'] = data['WEIGHT_SYSTOT'] * (data['WEIGHT_NOZ'] + data['WEIGHT_CP'] - 1.0)

    ## Prepare for S_LM
    # 1.Convert RA, DEC to theta, phi coord, 2.Convert from Angle to radian
    data_theta = numpy.pi / 2 - numpy.array(data['DEC']) / 180 * numpy.pi
    data_phi = numpy.array(data['RA']) / 180 * numpy.pi
    randoms_theta = numpy.pi / 2 - numpy.array(randoms['DEC']) / 180 * numpy.pi
    randoms_phi = numpy.array(randoms['RA']) / 180 * numpy.pi

    # # add y20_weight
    # data['y20'] = sph_harm(0, 2, data_phi, data_theta)
    # randoms['y20'] = sph_harm(0, 2, randoms_phi, randoms_theta)

    # =====================================================================================
    ## Catalog to mesh
    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(Nmesh=Nmesh[0], nbar='NZ', BoxSize=BoxSize, \
                       fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window=mass_assignment, interlaced=interlaced)
    rfield = mesh.compute()
    if not silent:
        print("Shape of FKP_field in this rank is", rfield.shape)

    ## Send sub_fields to rank = 0 process and concatenate them 
    send_data = numpy.array(rfield).astype("float64")
    recv_rf = comm.gather(send_data, root=0)

    if comm.rank == 0:
        res = numpy.concatenate(recv_rf, axis=1)
        store_npy_file = os.path.join(store_path, "FKP_fields", \
                                      "FKP_field_" + (data_file_list[i]).replace(".fits", ".npy"))
        numpy.save(store_npy_file, res)
        if not silent:
            print("I am Groot.")
            print("Shape of concatenated FKP_field is", res.shape)
            print(mesh.attrs)
        del res
        gc.collect()

    del rfield, recv_rf
    gc.collect()

    # =====================================================================================
    ## In order to get alpha
    # data_WEIGHT_sum = numpy.sum(numpy.array(data['WEIGHT'])*numpy.array(data['WEIGHT_FKP']))
    # randoms_WEIGHT_sum = numpy.sum(numpy.array(randoms['WEIGHT'])*numpy.array(randoms['WEIGHT_FKP']))
    data_WEIGHT_sum = numpy.sum(numpy.array(data['WEIGHT']))
    randoms_WEIGHT_sum = numpy.sum(numpy.array(randoms['WEIGHT']))

    # In order to get I33
    ## Note that numpy.array(randoms['WEIGHT'] = 1
    sub_I33 = numpy.sum(numpy.array(randoms['NZ']) ** 2 * numpy.array(randoms['WEIGHT_FKP']) ** 3)

    ## In order to get S_00
    data_S = numpy.sum((numpy.array(data['WEIGHT']) * numpy.array(data['WEIGHT_FKP'])) ** 3)
    randoms_S = numpy.sum((numpy.array(randoms['WEIGHT']) * numpy.array(randoms['WEIGHT_FKP'])) ** 3)

    ## In order to get S_LM(here we only want S_20)
    # data_S_LM = numpy.sum((numpy.array(data['WEIGHT']) \
    #                        * numpy.array(data['WEIGHT_FKP'])) ** 3 * numpy.array(data['y20']) ** 2)
    # randoms_S_LM = numpy.sum((numpy.array(randoms['WEIGHT']) \
    #                           * numpy.array(randoms['WEIGHT_FKP'])) ** 3 * (numpy.array(randoms['y20']) ** 2))

    ## In order to get norm of Nfield （DON'T mix it with I33）
    data_norm = numpy.sum((numpy.array(data['WEIGHT']) * numpy.array(data['WEIGHT_FKP'])) ** 2)
    randoms_norm = numpy.sum((numpy.array(randoms['WEIGHT']) * numpy.array(randoms['WEIGHT_FKP'])) ** 2)

    ## Send some attrs to root process
    send_attrs = numpy.array([data_WEIGHT_sum, randoms_WEIGHT_sum, sub_I33 \
                                 , data_S, randoms_S, data_norm, randoms_norm]).reshape(7, 1)
    recv_attrs = comm.gather(send_attrs, root=0)

    ## Concatenate the attrs and calculate some parameter needed to be stored
    if comm.rank == 0:
        attrs = numpy.concatenate(recv_attrs, axis=1)
        alpha = numpy.sum(attrs[0]) / numpy.sum(attrs[1])
        I33 = alpha * numpy.sum(attrs[2])
        S_00 = numpy.sum(attrs[3]) - alpha ** 3 * numpy.sum(attrs[4])
        # S_20 = 9 / 5 * (numpy.sum(attrs[5]) - alpha ** 3 * numpy.sum(attrs[6]))
        norm = numpy.sum(attrs[5]) + alpha ** 2 * numpy.sum(attrs[6])
        BoxCenter = mesh.attrs["BoxCenter"]

        if not silent:
            print("I am Groot.")
            print("attrs =", attrs)
            print("alpha =", alpha)
            print("I33  =", I33)
            print("S_00 =", S_00)
            # print("S_20 =", S_20)
            print("norm =", norm)
            print("BoxCenter =", BoxCenter)

        to_store = numpy.real(numpy.array([alpha, I33, S_00, norm, BoxCenter[0], BoxCenter[1], BoxCenter[2]]))
        store_txt_file = os.path.join(store_path, "attrs_stores",
                                      "info_of_" + (data_file_list[i]).replace("fits", "txt"))
        numpy.savetxt(store_txt_file, to_store)

        af_arr = alpha * numpy.ones(size)  ## send alpha back to each rank
    else:
        af_arr = None

    ## Prepare to get the N_field (Line2 of Eq.46 in https://arxiv.org/abs/1803.02132v3)
    af = comm.scatter(af_arr, root=0)
    BoxCenter = mesh.attrs["BoxCenter"]
    mymesh = (get_Ncatalog(data, randoms, BoxCenter, af)).to_mesh(Nmesh=Nmesh[0], \
                                                                  BoxSize=BoxSize, weight='Weight',
                                                                  resampler=mass_assignment, \
                                                                  interlaced=interlaced, dtype='complex128')
    Nfield = mymesh.compute()
    Nfield = numpy.array(Nfield).astype("f8")
    if not silent:
        print("af =", af)
        print("Nshape =", Nfield.shape)

    recv_nf = comm.gather(Nfield, root=0)

    if comm.rank == 0:
        res = numpy.concatenate(recv_nf, axis=1)
        store_npy_file = os.path.join(store_path, "N_fields", "N_field_" + (data_file_list[i]).replace(".fits", ".npy"))
        numpy.save(store_npy_file, res)
        if not silent:
            print(res.shape)
            print(mesh.attrs)

        del res
        gc.collect()

    del Nfield, recv_nf
    gc.collect()

    if comm.rank == 0:
        end_time = time.time()
        interval = end_time - start_time
        if not silent:
            print("Preprocessing time is %s s" % interval)
