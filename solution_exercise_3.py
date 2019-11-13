import uproot 
import numpy as np


import matplotlib.pyplot as plt



if __name__ == '__main__' :


    #open file
    signal_f = uproot.open('GluGluHToWWTo2L2Nu_M125.root')
    bkg_f = ( uproot.open('DYJetsToLL_M-10to50-LO.root'), uproot.open('DYJetsToLL_M-50-LO.root') )

    #open trees
    sig_tree = signal_f[ 'Events' ]
    bkg_trees = [ f['Events'] for f in bkg_f ]

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
