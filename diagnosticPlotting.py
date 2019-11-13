#copied from https://github.com/wverbeke/deepLearning/blob/master/diagnosticPlotting.py and slightly modified

#import necessary libraries 
import matplotlib.pyplot as plt
import numpy as np


def plotKerasMetricComparison( training_history, model_name, metric, metric_label ):

    #extract history of the metric 
    training_metric = training_history.history[metric]
    validation_metric = training_history.history['val_' + metric]

    #plot metric as a function of the training epoch
    epochs = range( 1, len(training_metric) + 1 )
    plt.plot(epochs, training_metric, 'b', label = 'training ' + metric_label.lower(), lw = 2 )
    plt.plot(epochs, validation_metric, 'r', label = 'validation ' + metric_label.lower(), lw = 2 )
    plt.legend( loc = 'best' )
    plt.xlabel( 'Epoch', fontsize = 22 )
    plt.ylabel( metric_label, fontsize = 22 )
    plt.ticklabel_format( style='sci', axis = 'y', scilimits = ( -2, 2 ) )
    plt.grid( True )
    plt.savefig( metric + '_' + model_name + '.pdf' )
    
    #clear canvas
    plt.clf()
    

def plotAccuracyComparison( training_history, model_name ):
    return plotKerasMetricComparison(training_history, model_name, 'acc', 'Accuracy')


def plotLossComparison( training_history, model_name ):
    return plotKerasMetricComparison(training_history, model_name, 'loss', 'Loss')


def computeEfficiency( outputs, weights, min_output, max_output, num_points ):
    cuts = min_output + np.arange(1, num_points + 1, dtype=float)/num_points*(  max_output - min_output )

    #reshape outputs to be one dimensional
    outputs = outputs.reshape( len(outputs) )
    denominator = np.sum( weights )

    #efficiency generator 
    efficiencies = ( np.sum( np.where( outputs > cuts[i], weights, 0 ) )/denominator for i in range( num_points ) )

    #initialize array from generator 
    efficiency_array = np.empty( num_points, dtype=float )
    for i, efficiency in enumerate( efficiencies ):
        efficiency_array[i] = efficiency

    return efficiency_array
        

def computeROC( outputs_signal, weights_signal, outputs_background, weights_background, num_points = 1000 ):

    min_output = min( np.min( outputs_signal ), np.min( outputs_background ) )
    max_output = max( np.max( outputs_signal ), np.max( outputs_background ) )
    
    #extra points to take into account 0% and 100% efficiency cases 
    sig_eff = np.zeros( num_points + 2 )
    sig_eff[0] = 1
    bkg_eff = np.zeros( num_points + 2 )
    bkg_eff[0] = 1

    #fill in intermediate points 
    sig_eff[1:num_points + 1] = computeEfficiency( outputs_signal, weights_signal, min_output, max_output, num_points )
    bkg_eff[1:num_points + 1] = computeEfficiency( outputs_background, weights_background, min_output, max_output, num_points )

    return sig_eff, bkg_eff
    
    
def backgroundRejection( bkg_eff ):
    bkg_rejection =  np.ones( len(bkg_eff) ) - bkg_eff
    return bkg_rejection


def plotROC(sig_eff, bkg_eff, model_name):
    
    #plot background rejection as a function of signal efficiency
    plt.plot( sig_eff, backgroundRejection(bkg_eff) , 'b', lw=2)
    plt.xlabel( 'Signal efficiency', fontsize = 22 )
    plt.ylabel( 'Background rejection', fontsize = 22 )
    plt.grid(True)
    plt.savefig('roc_' + model_name + '.pdf') 

    #clear canvas
    plt.clf()


#compute area under the ROC curve, a strong metric for evaluating the model performance
def areaUnderCurve(sig_eff, bkg_eff):
    
    #use trapezoidal rule to compute integral
    integral = np.trapz( backgroundRejection(bkg_eff), sig_eff)

    #integral could be negative depending on order of efficiency arrays (i.e. if it starts at high efficiency)
    return abs( integral )
    

def plotOutputShapeComparison( outputs_signal_training, weights_signal_training, 
    outputs_background_training, weights_background_training, 
    outputs_signal_testing, weights_signal_testing, 
    outputs_background_testing, weights_background_testing,
    model_name
    ):

    min_output = min( np.min(outputs_signal_training), np.min(outputs_background_training), np.min(outputs_signal_testing), np.min(outputs_background_testing ) )
    max_output = max( np.max(outputs_signal_training), np.max(outputs_background_training), np.max(outputs_signal_testing), np.max(outputs_background_testing ) )
    
    addHist( outputs_background_training, weights_background_training, 20, min_output, max_output, 'Background (training set)', color='red')
    addHist( outputs_background_testing, weights_background_testing, 20, min_output, max_output, 'Background (validation set)', color = 'purple')
    addHist( outputs_signal_training, weights_signal_training, 20, min_output, max_output, 'Signal (training set)', color='blue')
    addHist( outputs_signal_testing, weights_signal_testing, 20, min_output, max_output, 'Signal (validation set)', color='green')

    plt.xlabel( 'Model output', fontsize = 15 )
    plt.ylabel( 'Normalized events', fontsize = 15 )
    plt.legend(ncol=2, prop={'size': 9})

    #plt.yscale('log')

    bottom, top = plt.ylim()
    plt.ylim( 0,  top*1.2)
    plt.savefig('shapeComparison_' + model_name + '.pdf')
    plt.clf()
    

def binWidth(num_bins, min_bin, max_bin):
    return float( max_bin - min_bin) / num_bins


def addHist( data, weights, num_bins, min_bin, max_bin, label, color = 'blue'):
    bin_width = binWidth( num_bins, min_bin, max_bin )
    n, bins, _ = plt.hist(data, bins=num_bins, range=(min_bin, max_bin), weights = weights/np.sum(weights), label = label, histtype='step', lw=2, color=color)
    bin_errors = errors( data, weights, num_bins, min_bin, max_bin)
    bin_centers = 0.5*(bins[1:] + bins[:-1]) 
    plt.errorbar(bin_centers, n, yerr=bin_errors, fmt='none', ecolor=color)


def errors( data, weights, num_bins, min_bin, max_bin ):
    bin_errors = np.zeros( num_bins )
    bin_width = binWidth( num_bins, min_bin, max_bin )
    for i, entry in enumerate(data):
        bin_number = int( (entry - min_bin) // bin_width )
        if bin_number == num_bins :
            bin_number = num_bins - 1
        weight_entry =  weights[i]
        bin_errors[bin_number] += weight_entry*weight_entry
    bin_errors = (bin_errors ** 0.5)
    scale = np.sum(weights)
    bin_errors /= scale
    return bin_errors 
