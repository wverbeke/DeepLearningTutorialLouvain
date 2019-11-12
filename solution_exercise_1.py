import uproot
import numpy as np


if __name__ == '__main__':

    #open file 
    f = uproot.open( 'GluGluHToWWTo2L2Nu_M125.root' )

    #print trees in file
    print( f.keys() )

    #open Events tree and print branches 
    t = f['Events']
    print( t.keys() )

    #read 'mtw1' and 'mtw2', compute their mean and variance, and normalize the distributions
    mtw1_array = t.array( 'mtw1' )
    mean_mtw1 = np.mean( mtw1_array )
    std_mtw1 = np.std( mtw1_array )
    
    mtw1_array /= mean_mtw1
    mtw1_array -= std_mtw1
