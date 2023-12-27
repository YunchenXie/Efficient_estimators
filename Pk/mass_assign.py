import numpy
import time
from nbodykit.utils import timer
from nbodykit.lab import *
import os
from nbodykit import CurrentMPIComm

comm = CurrentMPIComm.get()
print("comm.rank=",comm.rank)


# Parameters setting
Nmesh = numpy.array([512,512,512])
BoxSize = numpy.array([6900,6900,6900]).astype("float64")

## many mock data with one random catalog!
data_file_list = []
file_source = "/public1/home/scg1018/share/Yunchen/desi_y1_blinded/data/QSO_NGC"
## randoms file
randoms_file = '/public1/home/scg1018/share/Yunchen/desi_y1_blinded/combined_randoms/combined_QSO_NGC_clustering.ran.fits'
## path to store npy and attrs files
store_path = os.path.join(os.getcwd(),"npy_stores")
attrs_path = os.path.join(os.getcwd(),"attrs_stores")## to store some attrs eg. N0,alpha...

if comm.rank == 0:
    print("I am Groot.")
    for item in [store_path,attrs_path]:
        if os.path.exists(item):
            pass
        else:
            os.mkdir(item)

## define a function to get file names in a chosen path
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
            # print(file_path)

listdir(file_source, data_file_list, True)
data_file_list.sort()

## In case you run this code on a Mac...Take care of the .DS_Store file
for item in data_file_list:
    if "Store" in item:
        data_file_list.remove(item)
        break
# print(data_file_list)


### Generating overdensity field from data/randoms catalogs
for i in range(1):
    start_time = time.time()

    # initialize the FITS catalog objects for data and randoms
    data = FITSCatalog(os.path.join(file_source,data_file_list[i]))
    randoms = FITSCatalog(randoms_file)

    # Select the Correct Redshift Range
    # for NGC
    ZMIN = 0.8
    ZMAX = 2.1

    # slice the randoms
    valid = (randoms['Z'] > ZMIN)&(randoms['Z'] < ZMAX)
    randoms = randoms[valid]

    # slice the data
    valid = (data['Z'] > ZMIN)&(data['Z'] < ZMAX)
    data = data[valid]

    # Adding the Cartesian Coordinates
    cosmo = cosmology.Cosmology(h=0.6736).match(Omega0_m=0.31377) ## planck 2018

    # add Cartesian position column
    data['Position'] = transform.SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
    randoms['Position'] = transform.SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)

    data["NZ"] = data["NX"]
    randoms["NZ"] = randoms["NX"]

    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(Nmesh=Nmesh[0], nbar = 'NZ', BoxSize=BoxSize,\
        fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc',interlaced = True)
        
    # 生成densityfield和对应的cfield
    rfield = mesh.compute()
    print(rfield.shape)

    send_data = numpy.array(rfield).astype("float64")
    recv_data = comm.gather(send_data, root=0)

    if comm.rank == 0:
        print("I am Groot.")
        res = numpy.concatenate(recv_data,axis=1)
        print(res.shape)
        # numpy.save("myres.npy",res)
        store_npy_file = os.path.join(store_path, "FKP_field_"+(data_file_list[i]).replace("fits","npy"))
        numpy.save(store_npy_file,res)
        print(mesh.attrs)



    data_WEIGHT_sum = numpy.sum(numpy.array(data['WEIGHT']))
    randoms_WEIGHT_sum = numpy.sum(numpy.array(randoms['WEIGHT']))

    sub_norm = (numpy.array(randoms['WEIGHT'])*numpy.array(randoms['NX'])\
        * numpy.array(randoms['WEIGHT_FKP'])*numpy.array(randoms['WEIGHT_FKP'])).sum()

    sub_N0_data = numpy.sum((numpy.array(data['WEIGHT'])* numpy.array(data['WEIGHT_FKP']))**2)
    sub_N0_randoms = numpy.sum((numpy.array(randoms['WEIGHT'])*numpy.array(randoms['WEIGHT_FKP']))**2)


    send_attrs = numpy.array([data_WEIGHT_sum,randoms_WEIGHT_sum,sub_norm,\
        sub_N0_data,sub_N0_randoms]).reshape(5,1)
    recv_attrs = comm.gather(send_attrs, root=0)


    if comm.rank == 0:
        print("I am Groot.")
        attrs = numpy.concatenate(recv_attrs,axis=1)
        alpha = numpy.sum(attrs[0])/numpy.sum(attrs[1])
        norm = alpha * (numpy.sum(attrs[2]))
        N0 = (numpy.sum(attrs[3]) + alpha**2 * numpy.sum(attrs[4]))/norm

        print("attrs =",attrs)
        print("alpha =",alpha)
        print("norm  =",norm)
        print("N0 =",N0)
        BoxCenter = mesh.attrs["BoxCenter"]
        print("BoxCenter =",BoxCenter)
        to_store = numpy.array([alpha,norm,N0,BoxCenter[0],BoxCenter[1],BoxCenter[2]])
        store_txt_file = os.path.join(attrs_path, "info_of_"+(data_file_list[i]).replace("fits","txt"))
        numpy.savetxt(store_txt_file,to_store)
        

        end_time = time.time()
        interval = end_time - start_time
        print("Preprocessing time is %s s"%interval)