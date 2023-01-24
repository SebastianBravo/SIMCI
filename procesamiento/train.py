import os
import torch
import numpy as np
from dataset_class import Dataset
from sklearn.model_selection import KFold

# Número de folds para k-fold cross-validation
num_folds = 10

# Cración objeto para validación cruzada
kfold = KFold(n_splits=num_folds, shuffle=True)

# Path de dataset
data_dir = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_no_scores_split/train'

# Extracción de path de cada item del dataset
data_files = []
for class_dir in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_dir):
        data_files += [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith('.npy')]
data_files = np.array(data_files)

# Parametros para generasión de datos
gen_param = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 4}

# Número de épocas
epochs = 1

# Lista para almacenar scores de cada fold
scores = []

# Ciclo de entrenamiento con validación cruzada
for train_index, val_index in kfold.split(data_files):
    # Para cada fold se define un dataset de entrenamiento por cada tipo de data augmentation
    train_set = Dataset(data_files[train_index], shape_aug=False, intensity_aug=False) # No augmentation
    train_set_shape = Dataset(data_files[train_index], shape_aug=True, intensity_aug=False) # Shape augmentation
    train_set_inte = Dataset(data_files[train_index], shape_aug=False, intensity_aug=True) # Intensity augmentation
    
    # Concatenar los 3 datasets de entrenamiento
    train_aug_set = torch.utils.data.ConcatDataset([train_set,train_set_shape,train_set_inte])
    
    # Para cada fold se define un dataset de validación
    val_set = Dataset(data_files[val_index], shape_aug=False, intensity_aug=False) # No augmentation
    
    # Dataloaders para entrenamiento y validación
    train_generator = torch.utils.data.DataLoader(train_aug_set, **gen_param)
    val_generator = torch.utils.data.DataLoader(val_set, **gen_param)
    
    # Ciclo de entrenamiento por épocas
    for epoch in range(epochs):
        for data, label in train_generator:
            print(label, flush=True)
            # forward 
            # Backward
            
    # Validación del sistema
