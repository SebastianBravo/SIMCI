import os
import data_generator
import numpy as np
import nibabel as nib
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
# model = load_model('outputs/modelo/3d_Unet_final.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})
model = load_model('outputs/checkpoints/model.49-0.97.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

# Data generator params
params = {'dim': (64,64,64),
          'batch_size': 4,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}

test_gen = data_generator.DataGenerator(path='F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data_split/test', **params)

# Evaluación
results = model.evaluate(test_gen)
