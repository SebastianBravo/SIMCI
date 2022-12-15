import os
import splitfolders
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from ttictoc import tic,toc
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Path Dataset
NFBS_Dataset_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/NFBS_Dataset'

# Path Imágenes corregidas
corrected_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Dataset_Corrected'

# Inicio timer pre procesamiento N4
# tic()

i = 0
'''---------------------Preprocesamiento de la imágenes---------------------'''
# N4 Bias Field Correction
for patient in os.listdir(NFBS_Dataset_path):
    # MRI T1 path
    mri_file = os.listdir(os.path.join(NFBS_Dataset_path,patient))[0]
    mri_path = os.path.join(NFBS_Dataset_path,patient,mri_file)
    
    # Lectura de MRI T1 formato nifti
    inputImage = sitk.ReadImage(mri_path, sitk.sitkFloat32)
    image = inputImage
    
    # N4 Bias Field Correction
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    
    # Guardado de imagen filtrada
    sitk.WriteImage(corrected_image_full_resolution, os.path.join(corrected_path,patient)+'.nii')

    i = i+1
    print(f'Listo {i}', flush=True)

# Tiempo n4: 27114 s = 7.5 h 

tic()

j = 0
# Volume Sampling
for i,patient in enumerate(os.listdir(NFBS_Dataset_path)):
    # Mask path
    mask_file = os.listdir(os.path.join(NFBS_Dataset_path,patient))[2]
    mask_path = os.path.join(NFBS_Dataset_path,patient,mask_file)
    
    # MRI path
    mri_path = os.path.join(corrected_path,(patient+'.nii'))
    
    # Importación máscara
    mask = nib.load(mask_path).get_fdata()
    mask = mask.astype(np.uint8)
    
    # Importación MRI
    corr_mri = nib.load(mri_path).get_fdata()
    
    # Normalización MRI
    corr_mri = (corr_mri-np.min(corr_mri))/(np.max(corr_mri)-np.min(corr_mri))
    
    # volume sampling
    mri_patches = patchify(corr_mri, (64, 64, 64), step=32)
    mask_patches = patchify(mask, (64, 64, 64), step=32)
    
    mri_patches = mri_patches.reshape(-1, mri_patches.shape[-3], mri_patches.shape[-2], mri_patches.shape[-1])
    mask_patches = mask_patches.reshape(-1, mask_patches.shape[-3], mask_patches.shape[-2], mask_patches.shape[-1])
    
    # Extender canales 
    mri = np.expand_dims(mri_patches, axis=4)
    mask = np.expand_dims(mask_patches, axis=4)
    
    for i in range(len(mri)):
        np.save(f'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data/images/{patient}_{i}.npy', mri[i])
        np.save(f'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data/masks/{patient}_{i}.npy', mask[i])
    
    j = j+1
    print(f'Listo {j}', flush=True)

print(toc())

# # Tiempo Normalización: 2265 s = 37.7 min

# Segmentación de conjuntos
splitfolders.ratio('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Data', output='Data_split', seed=1337, ratio=(0.70, 0.15, 0.15))
