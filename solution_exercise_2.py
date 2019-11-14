import uproot 
import numpy as np



if __name__ == '__main__' :

    #open files
    signal_f = uproot.open('GluGluHToWWTo2L2Nu_M125.root')
    bkg_f = ( uproot.open('DYJetsToLL_M-10to50-LO.root'), uproot.open('DYJetsToLL_M-50-LO.root') )

    #open trees
    sig_tree = signal_f[ 'Events' ]
    bkg_trees = [ f['Events'] for f in bkg_f ]

    #list of input variables
    input_variables = ['ptll', 'mth', 'uperp', 'upara', 'ptTOT_cut', 'mTOT_cut']

    #function to convert dictionary of arrays to 2D array
    def arrayDictTo2DArray( array_dict ):
        ret_array = None
        for key in array_dict:
            if ret_array is None:
                ret_array = array_dict[key]
                ret_array = np.expand_dims( ret_array, 1 )
            else :
                new_array = np.expand_dims( array_dict[key], 1 )
                ret_array = np.concatenate( [ret_array, new_array], axis = 1 )
        return ret_array


    #function to normalize array of inputs ( center distribution at 0 and give them unit variance )
    def normalizeInputArray( array2D ):

        #calculate mean along event axis, and expand dimensions of array so that it can be subtracted from previous array
        array2D -= np.expand_dims( np.mean( array2D, axis = 1 ), axis = 1 )
        array2D /= np.expand_dims( np.std( array2D, axis = 1 ), axis = 1 )
        return array2D

    #build signal and background arrays
    signal_data = normalizeInputArray( arrayDictTo2DArray( sig_tree.arrays( input_variables ) ) )
    background_data_list = [ normalizeInputArray( arrayDictTo2DArray(tree.arrays( input_variables ) ) ) for tree in bkg_trees ]

    #merge arrays for two different backgrounds
    background_data = np.concatenate( background_data_list, axis = 0 )
