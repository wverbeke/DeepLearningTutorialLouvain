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
        array2D -= np.mean( array2D, axis = 0 )
        array2D /= np.std( array2D, axis = 0 )
        return array2D


    #build signal and background arrays
    signal_data = normalizeInputArray( arrayDictTo2DArray( sig_tree.arrays( input_variables ) ) )
    background_data_list = [ normalizeInputArray( arrayDictTo2DArray(tree.arrays( input_variables ) ) ) for tree in bkg_trees ]

    #merge arrays for two different backgrounds
    background_data = np.concatenate( background_data_list, axis = 0 )


	#list of variables representing weights 
    weight_variables = [ 'XSWeight', 'SFweight2l', 'LepSF2l__ele_mvaFall17V1Iso_WP90__mu_cut_Tight_HWWW', 'LepCut2l__ele_mvaFall17V1Iso_WP90__mu_cut_Tight_HWWW', 'GenLepMatch2l', 'METFilter_MC' ]

    #function to build the total weight array for one tree 
    def getWeightArray( tree, weight_variables ):
        weight_array = None
        for key in weight_variables:
            if weight_array is None:
                weight_array = tree.array( key )
            else :
                weight_array *= tree.array( key )
        return weight_array

    #read signal and background weights
    signal_weights = getWeightArray( sig_tree, weight_variables )
    bkg_weights = np.concatenate( [ getWeightArray( bkg_tree, weight_variables ) for bkg_tree in bkg_trees ], axis = 0 )

    #avoid large numerical scales of weights
    signal_weights /= np.mean( signal_weights )
    bkg_weights /= np.mean( bkg_weights )

    #define arrays with labels for signal and background events
    #these are what the neural network will try to predict
    signal_labels = np.ones( len( signal_weights ) )
    bkg_labels = np.zeros( len( bkg_weights ) )


	#####################################
	# New code for exercise 4 starts here
	#####################################

    #make a randomly shuffled set of indices for given array
    def randomlyShuffledIndices( array ):
        indices = np.arange( len(array) )
        np.random.shuffle( indices )
        return indices
    
    #split array with given validation and test fraction
    def splitTrainValTest( array, validation_fraction = 0.4, test_fraction = 0.2 ):
        index_split_val =  int( ( 1 - validation_fraction - test_fraction )* len(array) )
        index_split_test = int( ( 1 - test_fraction ) * len(array) )
        return array[:index_split_val], array[index_split_val:index_split_test], array[index_split_test:]
    
    
    #simultaneously randomize inputs, weights and labels
    def randomize( data, labels, weights ) :
        random_indices = randomlyShuffledIndices( data )
        data = data[ random_indices ]
        labels = labels[ random_indices ]
        weights = weights[ random_indices ]
    
    #randomize signal and background events 
    randomize( signal_data, signal_labels, signal_weights )
    randomize( background_data, bkg_labels, bkg_weights )
    
    #split signal data
    signal_data_train, signal_data_val, signal_data_test = splitTrainValTest( signal_data )
    signal_labels_train, signal_labels_val, signal_labels_test = splitTrainValTest( signal_labels )
    signal_weights_train, signal_weights_val, signal_weights_test = splitTrainValTest( signal_weights )
    
    #split background data
    bkg_data_train, bkg_data_val, bkg_data_test = splitTrainValTest( background_data )
    bkg_labels_train, bkg_labels_val, bkg_labels_test = splitTrainValTest( bkg_labels )
    bkg_weights_train, bkg_weights_val, bkg_weights_test = splitTrainValTest( bkg_weights)
