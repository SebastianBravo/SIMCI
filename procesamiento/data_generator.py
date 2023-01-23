import os
import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical

class ClassificationDataGenerator(Sequence):
    def __init__(self, data_files, index, batch_size, class_mode='binary'):
        """
        Initialize the classification data generator.
        :param data_files: List of all data files
        :param index: List of indices to select the data
        :param batch_size: The batch size for the data generator.
        :param class_mode: The class mode for the data generator (binary or categorical).
        """
        self.data_files = np.array(data_files)[index]
        self.batch_size = batch_size
        self.class_mode = class_mode

    def __len__(self):
        """
        Number of batches per epoch
        """
        return len(self.data_files) // self.batch_size

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        batch_data = self.data_files[idx*self.batch_size:(idx+1)*self.batch_size]
        X = []
        y = []
        for data_file in batch_data:
            data = np.load(data_file)
            X.append(data)
            if self.class_mode == 'binary':
                class_name = os.path.basename(os.path.dirname(data_file))
                y.append(0 if class_name.startswith('class1') else 1)
            elif self.class_mode == 'categorical':
                class_name = os.path.basename(os.path.dirname(data_file))
                class_idx = 0 if class_name.startswith('class1') else 1
                y.append(to_categorical(class_idx, num_classes=2))
        return np.array(X), np.array(y)