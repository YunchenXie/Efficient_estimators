import numpy
from scipy.special import sph_harm
from sympy.physics.wigner import wigner_3j


##Some useful functions

"""
    Get distance of every mesh grid to the mesh-center in k-space
"""
def get_k_dist(Nmesh,k_mode):
    my_index = numpy.arange(Nmesh[0])
    my_index[Nmesh[0]//2:] -= Nmesh[0]
    my_ones = numpy.ones(Nmesh[0])
    arr1 = numpy.einsum("a,b,c->abc",my_index,my_ones,my_ones)
    arr2 = numpy.einsum("a,b,c->abc",my_ones,my_index,my_ones)
    arr3 = numpy.einsum("a,b,c->abc",my_ones,my_ones,my_index)
    return numpy.sqrt(arr1**2*(k_mode[0])**2+ arr2**2*(k_mode[1])**2+ arr3**2*(k_mode[2])**2)


"""
    Get direction of every mesh grid with respect to the the mesh-center in k-space
    # We note that when x=0, a singularity occurs in the calculation of phi, and that spherical harmonics are meaningless when x=y=z=0 
    # Thankfully, numpy has automatically circumvent these problems, but it is better to assign values to empty values to avoid errors 
    # numpy automatically sets the part divided by 0 to null 
    # Assign values to mesh grids x=y=z=0, making arrays complete and convenient for the following calculation
"""
def get_k_direction(Nmesh):
    my_ind = numpy.arange(Nmesh[0])
    my_ind[Nmesh[0]//2:] -= Nmesh[0]
    ones = numpy.ones(Nmesh[0])
    arr_x = numpy.einsum('a,b,c -> abc',my_ind,ones,ones)
    arr_y = numpy.einsum('a,b,c -> abc',ones,my_ind,ones)
    arr_z = numpy.einsum('a,b,c -> abc',ones,ones,my_ind)

    half_N = int(Nmesh[0]//2)

    arr_theta = numpy.arccos(arr_z/(numpy.sqrt(arr_x**2+arr_y**2+arr_z**2)))
    arr_theta[0,0,0] = 0

    arr_phi = numpy.arctan(arr_y/arr_x)
    arr_phi[half_N:,:,:] += numpy.pi ## For grids where x<0 should be added by Pi
    arr_phi[0,0,:] = 1 
    # These grids get a null value after the 0/0 divisions have on performed on them ; 
    # Grids on the z-axis can be arbitrarily specified because their corresponding sin(theta)=0
    #The above operation also gives an assignment to the point x=y=z=0,
    return arr_theta, arr_phi


"""
    Get distance of every mesh grid to the mesh-center in x-space (configuration space)
"""
def get_real_direction(Nmesh,BoxSize,BoxCenter):
    my_ind = numpy.arange(Nmesh[0]).astype("float64")
    my_ind[Nmesh[0]//2:] -= Nmesh[0]
    my_ind *= BoxSize[0]/Nmesh[0]
    ones = numpy.ones(Nmesh[0])
    arr_x = numpy.einsum('a,b,c -> abc',my_ind,ones,ones)
    arr_y = numpy.einsum('a,b,c -> abc',ones,my_ind,ones)
    arr_z = numpy.einsum('a,b,c -> abc',ones,ones,my_ind)

    #To correct the position +0.5*BoxSize/Nmesh + BoxCenter，This just avoids having a denominator of 0 when converting to theta,phi
    arr_x += 0.5*BoxSize[0]/Nmesh[0] + BoxCenter[0]
    arr_y += 0.5*BoxSize[1]/Nmesh[1] + BoxCenter[1]
    arr_z += 0.5*BoxSize[2]/Nmesh[2] + BoxCenter[2]

    # Switch to the spherical coordinate system
    arr_theta = numpy.arccos(arr_z/(numpy.sqrt(arr_x**2+arr_y**2+arr_z**2)))
    arr_phi = numpy.arctan(arr_y/arr_x)
    chosen = arr_x < 0
    arr_phi[chosen] += numpy.pi

    return arr_theta, arr_phi


"""
    space inversion, swap f(x) \to f(-x)
    Here we use an optimized version at Ruiyang's suggestion
    which is nearly 5 times faster than the original one!
"""
def space_inversion(f):
    arr = numpy.empty_like(f)
    arr[0, 0, 0] = f[0, 0, 0]
    arr[0, 0, 1:] = numpy.flip(f[0, 0, 1:])
    arr[0, 1:, 0] = numpy.flip(f[0, 1:, 0])
    arr[1:, 0, 0] = numpy.flip(f[1:, 0, 0])
    arr[0, 1:, 1:] = numpy.flip(f[0, 1:, 1:])
    arr[1:, 0, 1:] = numpy.flip(f[1:, 0, 1:])
    arr[1:, 1:, 0] = numpy.flip(f[1:, 1:, 0])
    arr[1:, 1:, 1:] = numpy.flip(f[1:, 1:, 1:])
    return arr
