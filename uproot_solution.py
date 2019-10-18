import uproot 
import numpy as np


import matplotlib.pyplot as plt



if __name__ == '__main__' :


    #open file
    f = uproot.open( 'tZq_2016.root' )

    #check all the objects in the file
    #print( f.keys() )

    #tree containing signal and background events 
    sig_tree =  f['signalTreeonZ_1bJet23Jets']
    bkg_tree =  f['backgroundTreeonZ_1bJet23Jets']

    #check the names of all branches in a tree
    #print( sig_tree.keys() )


    input_features = [ 'jetCSV1', 'jetCSV2', 'jetCSV3', 'jetCSV4', 'jetCSV5', 'jetCSV6', 'jetEta1', 'jetEta2', 'jetEta3', 'jetEta4', 'jetEta5', 'jetEta6', 'jetM1', 'jetM2', 'jetM3', 'jetM4', 'jetM5', 'jetM6', 'jetPhi1', 'jetPhi2', 'jetPhi3', 'jetPhi4', 'jetPhi5', 'jetPhi6', 'jetPt1', 'jetPt2', 'jetPt3', 'jetPt4', 'jetPt5', 'jetPt6', 'lepCharge1', 'lepCharge2', 'lepCharge3', 'lepEta1', 'lepEta2', 'lepEta3', 'lepFlavor1', 'lepFlavor2', 'lepFlavor3', 'lepPhi1', 'lepPhi2', 'lepPhi3', 'lepPt1', 'lepPt2', 'lepPt3', 'metPt', 'metPhi' ]

    #make array of one variable
    lepPt1 = sig_tree.array( 'lepPt1' ) 
    
    #or equivalent 
    sig_tree['lepPt1'].array() 

    #plot input variable
    def plotVar( tree, var_name ):
        array = tree.array( var_name )
        plt.hist( array, 100 )
        plt.xlabel( var_name )
        plt.ylabel( 'Events' )
        plt.savefig( tree.name.decode( encoding = 'UTF-8' ) + "_" + var + '.pdf' )
        plt.clf()


    ##plot signal distributions
    #for var in input_features:
    #    plotVar( sig_tree, var )


    ##plot background distributions
    #for var in input_featurs:   
    #    plotVar( bkg_tree, var )


    #make training / validation and test datasets 

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

    signal_dataset = arrayDictTo2DArray( sig_tree.arrays( input_features ) )
    signal_weights = sig_tree.array('eventWeight')
    background_dataset = arrayDictTo2DArray( bkg_tree.arrays( input_features ) )
    background_weights = bkg_tree.array('eventWeight')

    print( signal_dataset.shape )

	
    #make training, test, and validation sets 
    def splitTrainValTest( array, validation_fraction = 0.4, test_fraction = 0.2 ):
        index_split_val =  int( validation_fraction * len(array) )
        index_split_test = int( ( 1 - test_fraction ) * len(array) )
        return array[:index_split_val], array[index_split_val:index_split_test], array[index_split_test:]
    

    def randomlyShuffledIndices( array ):
        indices = np.arange( len(array) )
        np.random.shuffle( indices )
        return indices 

    
    def mergeAndShuffle( signal_data, background_data, signal_weights = None, background_weights = None):
        merged_set = np.concatenate( [signal_data, background_data], axis = 0 )
        merged_labels = np.concatenate( [np.ones( len( signal_data ) ), np.zeros( len(background_data) )], axis = 0 )
        merged_weights = None
        if signal_weights is not None and background_weights is not None:
        	merged_weights = np.concatenate( [signal_weights, background_weights] , axis = 0 )
        
        random_indices = randomlyShuffledIndices( merged_set )
        merged_set = merged_set[ random_indices ]
        merged_labels = merged_labels[ random_indices ]
        if merged_weights is not None:
        	merged_weights = merged_weights[ random_indices ]
        
        if merged_weights is None:
        	return ( merged_set, merged_labels )
        else :
        	return ( merged_set, merged_labels, merged_weights )

    dataset, labels, weights = mergeAndShuffle( signal_dataset, background_dataset, signal_weights, background_weights )


    train_set, val_set, test_set = splitTrainValTest( dataset )
    train_labels, val_labels, test_labels = splitTrainValTest( labels )
    train_weights, val_weights, test_weights = splitTrainValTest( weights )

    print( train_set.shape )

    from keras import models
    from keras import layers
    from keras import optimizers
    from keras import losses
    
    network = models.Sequential()
    network.add( layers.Dense(128, activation='relu', input_shape=( len( input_features ), ) ) )
    network.add( layers.Dense(128, activation='relu' ) )
    network.add( layers.Dense(128, activation='relu' ) )
    network.add( layers.Dense(128, activation='relu' ) )
    network.add( layers.Dense(128, activation='relu' ) )
    network.add( layers.Dense(1, activation='sigmoid' ) )


    #to use auc as a keras metric 
    import tensorflow as tf
    
    network.compile(
        optimizer=optimizers.Nadam(),
        loss=losses.binary_crossentropy,
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    history = network.fit(
        train_set,
        train_labels,
        #sample_weight = None,
        epochs=40,
        batch_size=512,
        validation_data=( val_set, val_labels ),
        verbose = 1
    )
