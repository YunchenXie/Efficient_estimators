import numpy
import os

mydict = {}
mydict["silent"] = False
mydict["Nmesh"] = numpy.array([512,512,512])
mydict["BoxSize"] = numpy.array([2000, 2000, 2000])
mydict["data_fits_path"] = "/public1/home/scg1018/share/Yunchen/desi_y1_blinded/data/BGS_SGC"
mydict["random_catalog_file"] = '/public1/home/scg1018/share/Yunchen/desi_y1_blinded/randoms/BGS_SGC/BGS_BRIGHT-21.5_SGC_0_clustering.ran.fits'
mydict['npy_store_path'] = os.path.join(os.getcwd(),"npy_stores") 
## we stored npys in current path default, you may change it if necessary!

mydict["z_range"] = [0.1,0.4]
mydict["cosmo_h"] = 0.6736
mydict["cosmo_Omega0"] = 0.31377
# mydict["cosmo_Omega0_b"] = 0.048
# mydict["cosmo_Omega0_cdm"] = 0.259115
# mydict["cosmo_n_s"] = 0.96
# mydict["cosmo_sigma8"] = 0.8288
mydict['mass_assignment'] = "tsc"
mydict["number_of_kmode"] = 6
mydict["k_num"] = 20 
mydict["interlaced"] = True
mydict["index_range"] = [0, 1]
## e.g. you can appoint mydict["index_range"] = [100, 150], then the mass_interpolation code will deal with number 101-150 mock catalogs


# mydict["FKP_source"] = os.path.join(os.getcwd(),"npy_stores/FKP_fields")
# mydict["N_source"] = os.path.join(os.getcwd(),"npy_stores/N_fields")
# mydict["attrs_source"] = os.path.join(os.getcwd(),"npy_stores/attrs_stores")
# mydict["Y_lm_arrays"] = os.path.join(os.getcwd(),"Y_lm_arrays")
mydict["res_collection"] = os.path.join(os.getcwd(),"Results_Collection")

numpy.save('settings.npy', mydict)
