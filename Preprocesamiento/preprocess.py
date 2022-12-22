import os
import numpy as np
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from patchify import patchify, unpatchify
from scipy import ndimage as ndi
from skimage import morphology
from ttictoc import tic,toc

"""-------------------------Importación modelo CNN-------------------------"""
# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
model = load_model('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/outputs/checkpoints/model.49-0.97.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

"""-------------------Extracción de cerebro para cada MRI------------------"""
# Path imágenes MRI
# mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/dataset_n4_all_scores'
# mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/dataset_n4_no_scores'
mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data no loaded'

# Path imágenes cerebro
brain_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/dataset_brain_no_scores'

# Tiempo inicio
tic()

# lista fallos
error_mri = []

idx = 0
for patient in os.listdir(mri_path):
    try:
        # Cargar imagen MRI
        mri_image = np.load(os.path.join(mri_path,patient))
        mri_image = np.transpose(mri_image)
        mri_image = np.append(mri_image, np.zeros((192-mri_image.shape[0],256,256,)), axis=0)
        
        # Preprocesamiento --------------------------------------------------------
        # Rotación
        mri_image = mri_image.astype(np.float32)
        mri_image = np.rot90(mri_image, axes=(1,2))
        
        # Volume sampling
        mri_patches = patchify(mri_image, (64, 64, 64), step=64)
        
        # Predicción de máscara de cerebro ----------------------------------------
        # Máscara de cerebro para cada volúmen
        mask_patches = []
        
        for i in range(mri_patches.shape[0]):
          for j in range(mri_patches.shape[1]):
            for k in range(mri_patches.shape[2]):
               single_patch = np.expand_dims(mri_patches[i,j,k,:,:,:], axis=0)
               single_patch_prediction = model.predict(single_patch, verbose=0)
               single_patch_prediction_th = (single_patch_prediction[0,:,:,:,0] > 0.5).astype(np.uint8)
               mask_patches.append(single_patch_prediction_th)
        
        # Conversión a numpy array
        predicted_patches = np.array(mask_patches)
        
        # Reshape para proceso de reconstrucción
        predicted_patches_reshaped = np.reshape(predicted_patches, 
                                                (mri_patches.shape[0], mri_patches.shape[1], mri_patches.shape[2],
                                                 mri_patches.shape[3], mri_patches.shape[4], mri_patches.shape[5]) )
        
        # Reconstrucción máscara
        reconstructed_mask = unpatchify(predicted_patches_reshaped, mri_image.shape)
        
        # Suavizado máscara
        corrected_mask = ndi.binary_closing(reconstructed_mask, structure=morphology.ball(2)).astype(np.uint8)
        
        # Eliminación de volumenes ruido
        no_noise_mask = corrected_mask.copy()
        mask_labeled = morphology.label(corrected_mask, background=0, connectivity=3)
        label_count = np.unique(mask_labeled, return_counts=True)
        brain_label = np.argmax(label_count[1][1:]) + 1
        
        no_noise_mask[np.where(mask_labeled != brain_label)] = 0
        
        # Elimicación huecos y hendiduras
        filled_mask = ndi.binary_closing(no_noise_mask, structure=morphology.ball(12)).astype(np.uint8)
        
        # Extracción de cerebro ---------------------------------------------------
        # Aplicar máscara a imagen mri
        mri_brain = np.multiply(mri_image,filled_mask)
        
        # Guardado de imagen ------------------------------------------------------
        out_path = os.path.join(brain_path,patient)
        np.save(out_path, mri_brain)
        
        # Conteo de imágenes guardadas
        print(f'Terminó la extracción de cerebro de MRI: {idx}', flush=True)
        idx = idx+1
        
    except:
        # imagenes con error
        print(f'Falló la extracción de cerebro de MRI: {idx} con nombre {patient}', flush=True)
        error_mri.append(patient)
        idx = idx+1
    
# Tiempo final
print(f'La extracción tuvo un tiempo de: {toc()} segundos')
print(f'Los siguientes MRI tuvieron fallos: {error_mri}')

# La extracción tuvo un tiempo de: 66033.26728 segundos (todos los scores)
# La extracción tuvo un tiempo de: 68908.6059376 segundos (no scores)

# Fallos en S62838.npy, S75197.npy, S66833.npy, S84065.npy, S34918.npy (todos los scores)
# Fallos en S18441.npy, S29082.npy (no scores)