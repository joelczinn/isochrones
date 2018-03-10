from numba import jit, float64
from math import sqrt
import numpy as np

#@jit(nopython=False)
def interp_box(x, y, z, box, values):
    """
    box is 8x3 array, though not really a box
    
    values is length-8 array, corresponding to values at the "box" coords

    TODO: should make power `p` an argument
    """
    
    # Calculate the distance to each vertex
    
    val = 0
    norm = 0
    for i in range(8):
        # JCZ 100118
        # re-implementing by defining the distance so if it happens to be the same as a grid point (zero distance), it doesn't fail.
        d = sqrt((x-box[i,0])**2 + (y-box[i,1])**2 + (z-box[i, 2])**2)
        if d == 0.0:
            w = 1.0
        else:
            # JCZ 300118
            # trying 1/2 distance as weight instead of just 1./d.
            w = 1./d**(0.5)
        val += w * values[i]
        norm += w

        # # Inv distance, or Inv-dsq weighting
        # w = 1./sqrt((x-box[i,0])**2 + (y-box[i,1])**2 + (z-box[i, 2])**2)
        # # w = 1./((x-box[i,0])*(x-box[i,0]) + 
        # #         (y-box[i,1])*(y-box[i,1]) + 
        # #         (z-box[i, 2])*(z-box[i, 2]))
        # val += w * values[i]
        # norm += w

    return val/norm

#@jit(nopython=False)
def searchsorted(arr, N, x):
    """N is length of arr
    """
    L = 0
    R = N-2
    done = False
    # JCZ 100118
    # adding equal
    equal = False
    m = (L+R)//2
    while not done:

        if arr[m] < x:
            L = m + 1
        elif arr[m] > x:
            R = m - 1
        elif arr[m] == x:
            done = True
            # JCZ 100118
            # adding equal
            equal = True
        m = (L+R)//2
        if L>R:
            done = True
    # JCZ 100118
    # this should actually be m... !!! JCZ 100118 RIGHT???
    # return L
    # returning equal, too
    if equal:
        return m, equal
    else:
        return L, equal
        
#@jit(nopython=False)
def searchsorted_many(arr, values):
    N = len(arr)
    Nval = len(values)
    inds = np.zeros(Nval)
    equals = np.zeros(Nval)
    for i in range(Nval):
        x = values[i]
        L = 0
        R = N-1
        done = False
        # JCZ 100118
        equal = False
        m = (L+R)//2
        while not done:
            if arr[m] < x:
                L = m + 1
            elif arr[m] > x:
                R = m - 1
            m = (L+R)//2
            if L>R:
                done = True
        # JCZ 100118
        # this should actually be m... !!! JCZ 100118 RIGHT???
        # inds[i] = L
        if equal:
            inds[i] = m
            equals[i] = equal
        else:
            inds[i] = L
            equals[i] = equal
    return inds, equals

#@jit(nopython=False)
def interp_values(mass_arr, age_arr, feh_arr, icol, 
                 grid, mass_col, ages, fehs, grid_Ns):
    """mass_arr, age_arr, feh_arr are all arrays at which values are desired

    icol is the column index of desired value
    grid is nfeh x nage x max(nmass) x ncols array
    mass_col is the column index of mass
    ages is grid of ages
    fehs is grid of fehs
    grid_Ns keeps track of nmass in each slice (beyond this are nans)
    
    """

    N = len(mass_arr)
    results = np.zeros(N)

    Nage = len(ages)
    Nfeh = len(fehs)

    for i in range(N):
        results[i] = interp_value(mass_arr[i], age_arr[i], feh_arr[i], icol, 
                                 grid, mass_col, ages, fehs, grid_Ns, False)

        ## Things are slightly faster if the below is used, but for consistency,
        ## using above.
        # mass = mass_arr[i]
        # age = age_arr[i]
        # feh = feh_arr[i]

        # ifeh = searchsorted(fehs, Nfeh, feh)
        # iage = searchsorted(ages, Nage, age)
        # if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
        #     results[i] = np.nan
        #     continue

        # pts = np.zeros((8,3))
        # vals = np.zeros(8)

        # i_f = ifeh - 1
        # i_a = iage - 1
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[0, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[0, 1] = ages[i_a]
        # pts[0, 2] = fehs[i_f]
        # vals[0] = grid[i_f, i_a, imass, icol]
        # pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[1, 1] = ages[i_a]
        # pts[1, 2] = fehs[i_f]
        # vals[1] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh - 1
        # i_a = iage 
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[2, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[2, 1] = ages[i_a]
        # pts[2, 2] = fehs[i_f]
        # vals[2] = grid[i_f, i_a, imass, icol]
        # pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[3, 1] = ages[i_a]
        # pts[3, 2] = fehs[i_f]
        # vals[3] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh
        # i_a = iage - 1
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[4, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[4, 1] = ages[i_a]
        # pts[4, 2] = fehs[i_f]
        # vals[4] = grid[i_f, i_a, imass, icol]
        # pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[5, 1] = ages[i_a]
        # pts[5, 2] = fehs[i_f]
        # vals[5] = grid[i_f, i_a, imass-1, icol]

        # i_f = ifeh 
        # i_a = iage
        # Nmass = grid_Ns[i_f, i_a]
        # imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
        # pts[6, 0] = grid[i_f, i_a, imass, mass_col]
        # pts[6, 1] = ages[i_a]
        # pts[6, 2] = fehs[i_f]
        # vals[6] = grid[i_f, i_a, imass, icol]
        # pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
        # pts[7, 1] = ages[i_a]
        # pts[7, 2] = fehs[i_f]
        # vals[7] = grid[i_f, i_a, imass-1, icol]
        
        # results[i] = interp_box(mass, age, feh, pts, vals)
        
    return results

#@jit(nopython=False)
def interp_value(mass, age, feh, icol, 
                 grid, mass_col, ages, fehs, grid_Ns, debug):
                 # return_box):
    """mass, age, feh are *single values* at which values are desired

    icol is the column index of desired value
    grid is nfeh x nage x max(nmass) x ncols array
    mass_col is the column index of mass
    ages is grid of ages
    fehs is grid of fehs
    grid_Ns keeps track of nmass in each slice (beyond this are nans)
    
    TODO:  fix situation where there is exact match in age, feh, so we just
    interpolate along the track, not between...
    """
    

    Nage = len(ages)
    Nfeh = len(fehs)
    # JCZ 100118
    # adding all these equal variables, below whenever there is a searchsorted call.
    ifeh, eqfeh = searchsorted(fehs, Nfeh, feh)
    iage, eqage = searchsorted(ages, Nage, age)
    # JCZ 100118
    # replacing the nan condition to be if it is off the grid.
    if feh < fehs[0] or feh > fehs[Nfeh-1] or age < ages[0] or age > ages[Nage-1]:
        return np.nan
    # if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
        # return np.nan

    pts = np.zeros((8,3))
    vals = np.zeros(8)


    if eqage:
        iage_m = iage
    else:
        iage_m = iage - 1

    if eqfeh:
        ifeh_m = ifeh
    else:
        ifeh_m = ifeh - 1
        
    i_f = ifeh_m
    i_a = iage_m
    Nmass = grid_Ns[i_f, i_a]
    # JCZ 150118
    # !!! something about this is wrong because i shouldn't have to do this test. that's the whole points
    # of the Nmass grid to make sure not overstepping the masses.......
    mass_max = Nmass
    mass_min = 0
        
    imass, eqmass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    # print imass
    # print grid[i_f, i_a, :, mass_col]
    # print eqmass
    # JCZ 100118
    if eqmass:
        imass_m = imass
    else:
        imass_m = imass - 1
    # print mass
    # print grid.shape
    # print imass, imass_m
    if imass > mass_max:
        vals[0] = np.nan
    else:
        pts[0, 0] = grid[i_f, i_a, imass, mass_col]
        pts[0, 1] = ages[i_a]
        pts[0, 2] = fehs[i_f]
        vals[0] = grid[i_f, i_a, imass, icol]
        
    if imass_m < mass_min:
        vals[1] = np.nan
    else:
        pts[1, 0] = grid[i_f, i_a, imass_m, mass_col]
        pts[1, 1] = ages[i_a]
        pts[1, 2] = fehs[i_f]
        vals[1] = grid[i_f, i_a, imass_m, icol]


    i_f = ifeh_m
    i_a = iage 
    Nmass = grid_Ns[i_f, i_a]
    imass, eqmass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    # JCZ 100118
    if eqmass:
        imass_m = imass
    else:
        imass_m = imass - 1
    pts[2, 0] = grid[i_f, i_a, imass, mass_col]
    pts[2, 1] = ages[i_a]
    pts[2, 2] = fehs[i_f]
    vals[2] = grid[i_f, i_a, imass, icol]
    pts[3, 0] = grid[i_f, i_a, imass_m, mass_col]
    pts[3, 1] = ages[i_a]
    pts[3, 2] = fehs[i_f]
    vals[3] = grid[i_f, i_a, imass_m, icol]

    i_f = ifeh
    i_a = iage_m
    Nmass = grid_Ns[i_f, i_a]
    imass, eqmass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    # JCZ 100118
    if eqmass:
        imass_m = imass
    else:
        imass_m = imass - 1
    
    pts[4, 0] = grid[i_f, i_a, imass, mass_col]
    pts[4, 1] = ages[i_a]
    pts[4, 2] = fehs[i_f]
    vals[4] = grid[i_f, i_a, imass, icol]

    pts[5, 0] = grid[i_f, i_a, imass_m, mass_col]
    pts[5, 1] = ages[i_a]
    pts[5, 2] = fehs[i_f]
    vals[5] = grid[i_f, i_a, imass_m, icol]

    i_f = ifeh 
    i_a = iage
    Nmass = grid_Ns[i_f, i_a]
    imass, eqmass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
    # JCZ 100118
    if eqmass:
        imass_m = imass
    else:
        imass_m = imass - 1
    
    pts[6, 0] = grid[i_f, i_a, imass, mass_col]
    pts[6, 1] = ages[i_a]
    pts[6, 2] = fehs[i_f]
    vals[6] = grid[i_f, i_a, imass, icol]
    pts[7, 0] = grid[i_f, i_a, imass_m, mass_col]
    pts[7, 1] = ages[i_a]
    pts[7, 2] = fehs[i_f]
    vals[7] = grid[i_f, i_a, imass_m, icol]
    
    # if debug:
    #     result = np.zeros((8,4))
    #     for i in range(8):
    #         result[i, 0] = pts[i, 0]
    #         result[i, 1] = pts[i, 1]
    #         result[i, 2] = pts[i, 2]
    #         result[i, 3] = vals[i]
    #     return result
    # else:

    return interp_box(mass, age, feh, pts, vals)

