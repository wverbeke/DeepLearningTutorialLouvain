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
    input_variables = [	'ptll', 'mth', 'jetpt1_cut', 'uperp', 'upara', 'PfMetDivSumMet', 'recoil', 'mpmet', 'mtw1', 'mtw2', 'PuppiMET_pt', 'MET_pt', 'TkMET_pt', 'projtkmet', 'projpfmet', 'dphilljet_cut', 'dphijet1met_cut', 'dphillmet', 'dphilmet1', 'dphilmet2', 'jetpt2_cut', 'dphijet2met_cut', 'dphilljetjet_cut', 'dphijjmet_cut', 'ptTOT_cut', 'mTOT_cut', 'PV_npvsGood' ]

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

    #make a randomly shuffled set of indices for given array
    def randomlyShuffledIndices( array ):
        indices = np.arange( len(array) )
        np.random.shuffle( indices )
        return indices
    
    #split array with given validation and test fraction
    def splitTrainValTest( array, validation_fraction = 0.4, test_fraction = 0.2 ):
        index_split_val =  int( validation_fraction * len(array) )
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

	#####################################
	# New code for exercise 5 starts here
	#####################################

    #merge signal and background datasets and shuffle them again for training
    def mergeAndShuffle( signal_data, background_data, signal_labels, background_labels, signal_weights, background_weights):
    	merged_set = np.concatenate( [signal_data, background_data], axis = 0 )
    	merged_labels = np.concatenate( [signal_labels, background_labels], axis = 0 )
    	merged_weights = np.concatenate( [signal_weights, background_weights] , axis = 0 )
    	
    	random_indices = randomlyShuffledIndices( merged_set )
    	merged_set = merged_set[ random_indices ]
    	merged_labels = merged_labels[ random_indices ]
    	merged_weights = merged_weights[ random_indices ]
    
    	return ( merged_set, merged_labels, merged_weights )
    
    
    #make merged training, validation and test datasets
    train_data, train_labels, train_weights = mergeAndShuffle( signal_data_train, bkg_data_train, signal_labels_train, bkg_labels_train, signal_weights_train, bkg_weights_train )
    val_data, val_labels, val_weights = mergeAndShuffle( signal_data_val, bkg_data_val, signal_labels_val, bkg_labels_val, signal_weights_val, bkg_weights_val )
    test_data, test_labels, test_weights = mergeAndShuffle( signal_data_test, bkg_data_test, signal_labels_test, bkg_labels_test, signal_weights_test, bkg_weights_test ) 
    
    #Now we have the datasets we need for training a neural network!
    #Given below is a ( currently VERY VERY BAD!!!) neural network that will be trained
    
    #keras modules
    from keras import models
    from keras import layers
    from keras import optimizers
    from keras import losses
    
    network = models.Sequential()
    network.add( layers.Dense(64, activation='linear' ) )
    network.add( layers.Dense(1, activation='sigmoid' ) )
    
    #to use auc as a keras metric 
    import tensorflow as tf
    
    network.compile(
        optimizer=optimizers.Nadam(),
        loss=losses.binary_crossentropy,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    history = network.fit(
        train_data,
        train_labels,
        sample_weight =train_weights,
        epochs=10,
        batch_size=1024,
        validation_data=( val_data, val_labels, val_weights ),
        verbose = 1
    )
    
    train_output_signal = network.predict( signal_data_train )
    train_output_bkg = network.predict( bkg_data_train ) 
    val_output_signal = network.predict( signal_data_val )
    val_output_bkg = network.predict( bkg_data_val )
    
    
    #plot ROC curve and compute AUC
    from diagnosticPlotting import plotKerasMetricComparison, plotROC, computeROC, plotOutputShapeComparison, areaUnderCurve
    
    sig_eff, bkg_eff = computeROC( val_output_signal, signal_weights_val, val_output_bkg, bkg_weights_val, num_points = 10000 )
    
    plotROC( sig_eff, bkg_eff, 'roc' )
    print('######################################' )
    roc_integral = areaUnderCurve( sig_eff, bkg_eff )
    print( 'ROC INTEGRAL = {}'.format( roc_integral ) )
    print('######################################' )
    
    #plot output shape for training and validation sets
    plotOutputShapeComparison( train_output_signal, signal_weights_train, train_output_bkg, bkg_weights_train,
        val_output_signal, signal_weights_val,
        val_output_bkg, bkg_weights_val,
        'model'
    )

