import numpy as np
from keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt 


def generateRandomDataset( size = 1000 ):
    random_array = np.random.uniform(0, 10, size )
    exponential_random_array = np.exp( random_array )
    return random_array, exponential_random_array


if __name__ == '__main__':
    samples, targets = generateRandomDataset( 100000 )

    validation_fraction = 0.1
    split_index = int( len( samples )*( 1 - validation_fraction ) )
    train_samples = samples[ : split_index ]
    train_targets = targets[ : split_index ]
    val_samples = samples[ split_index : ]
    val_targets = targets[ split_index : ]

    nodes_per_layer = 64

    ###############################
    #Change activation function here to see the effect
    ###############################
    #activation = 'linear'
    activation = 'relu'

    model = models.Sequential()
    model.add( layers.Dense( nodes_per_layer, activation = activation, input_shape = ( 1, ) ) )
    model.add( layers.Dense( nodes_per_layer, activation = activation ) )
    model.add( layers.Dense( nodes_per_layer, activation = activation ) )
    model.add( layers.Dense( nodes_per_layer, activation = activation ) )
    model.add( layers.Dense( nodes_per_layer, activation = activation ) )
    model.add( layers.Dense( 1, activation = 'linear' ) )

    model.compile( 
        optimizer=optimizers.Adam(),
        loss=losses.mean_squared_error
    )


    history = model.fit(
        train_samples,
        train_targets,
        epochs = 20,
        batch_size = 512,
        validation_data = ( val_samples, val_targets ),
        verbose = 1
    ) 


    points = np.arange( 0, 9.9, 0.005 )
    true = np.exp( points )
    predictions = model.predict( points )

    plt.plot( points, true, 'r', label = 'true' )
    plt.plot( points, predictions, 'b', label = 'predictions' )
    plt.xlabel( 'x', fontsize = 15 )
    plt.ylabel( 'exp( x )', fontsize = 15 )
    plt.legend( fontsize = 15)
    plt.savefig('exptest.pdf')
