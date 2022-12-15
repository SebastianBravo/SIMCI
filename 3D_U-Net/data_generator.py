import numpy as np
import keras
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size=32, dim=(64,64,64), n_channels=1,
                 n_classes=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.path = path
        self.batch_size = batch_size
        self.list_IDs = os.listdir(path+'/images')
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(self.path + '/images/' + ID).astype(np.float32)

            # Store class
            y[i] = np.load(self.path + '/masks/' + ID).astype(np.float32)

        return X, y