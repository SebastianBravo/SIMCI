import os
import numpy as np
import nibabel as nib
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from matplotlib.widgets import Slider
from scipy import ndimage as ndi
from skimage import morphology
import cv2

"""-------------------------Importación modelo CNN-------------------------"""
# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
model = load_model('outputs/modelo/3d_Unet_final.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

"""-------------------------Importación imagen MRI-------------------------"""
# Path imágenes MRI
mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/3D_U-Net/Dataset_Corrected'

# Paciente aleatorio
img_idx = np.random.randint(0,125)
patient = os.listdir(mri_path)[img_idx]

# Cargar imagen MRI
mri_image = nib.load(os.path.join(mri_path,patient)).get_fdata()

"""----------------------Preprocesamiento imagen MRI-----------------------"""
# Rotación
mri_image = mri_image.astype(np.float32)
mri_image = (mri_image-np.min(mri_image))/(np.max(mri_image)-np.min(mri_image))

# Volume sampling
mri_patches = patchify(mri_image, (64, 64, 64), step=64)

"""--------------------Predicción de máscara de cerebro--------------------"""
# Máscara de cerebro para cada volúmen
mask_patches = []

for i in range(mri_patches.shape[0]):
  for j in range(mri_patches.shape[1]):
    for k in range(mri_patches.shape[2]):
      single_patch = np.expand_dims(mri_patches[i,j,k,:,:,:], axis=0)
      single_patch_prediction = model.predict(single_patch)
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

# Rellenar huecos
# filled_mask = ndi.binary_closing(reconstructed_mask, structure=np.ones((17,17,17)))
filled_mask = ndi.binary_closing(reconstructed_mask, structure=morphology.ball(9))

"""-------------------------Extracción de cerebro--------------------------"""
# Aplicar máscara a imagen mri
mri_brain = np.multiply(mri_image,reconstructed_mask)
mri_brain_filled = np.multiply(mri_image,filled_mask)

"""------------------------Visualización resultados------------------------"""
# Visualización resultados 
mri_slice = 100

# Plot Comparación máscaras
fig, axs = plt.subplots(1,3)
fig.subplots_adjust(bottom=0.15)
fig.suptitle('Comparación Máscaras Obtenidas')

axs[0].set_title('MRI sin máscara')
axs[0].imshow(mri_image[:,:,mri_slice],cmap='gray')

axs[1].set_title('MRI aplicando máscara 3D U-Net')
axs[1].imshow(mri_brain[:,:,mri_slice],cmap='gray')

axs[2].set_title('MRI aplicando máscara 3D U-Net corregida')
axs[2].imshow(mri_brain_filled[:,:,mri_slice],cmap='gray')

# Slider para cambiar slice
ax_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
mri_slice_slider = Slider(ax_slider, 'Slice', 0, 192, 100, valstep=1)

def update(val):
    mri_slice = mri_slice_slider.val
    
    axs[0].imshow(mri_image[:,:,mri_slice],cmap='gray')
    axs[1].imshow(mri_brain[:,:,mri_slice],cmap='gray')
    axs[2].imshow(mri_brain_filled[:,:,mri_slice],cmap='gray')
    
# Actualizar plot comparación máscaras
mri_slice_slider.on_changed(update)

# Plot comparación de contornos
fig2, axs2 = plt.subplots()
fig2.subplots_adjust(bottom=0.15)
axs2.set_title('Comparación Contornos de Máscaras')

# Mri
axs2.imshow(mri_image[:,:,mri_slice],cmap='gray', vmin=0, vmax=np.max(mri_image))

# Contornos
mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice], levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linewidths=1.5)
filled_mask_cocontour = axs2.contour(filled_mask[:,:,mri_slice], levels=np.logspace(-4.7, -3., 10), colors='#fe7500', linestyles='dotted', linewidths=1.5)

# out_mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)
# dilated_mask_contour = axs2.contour(dilated_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)

# Etiquetas
h1,_ = mask_contour.legend_elements()
h2,_ = filled_mask_cocontour.legend_elements()

axs2.legend([h1[0], h2[0]], ['3D U-Net', '3D U-Net Corregida'],facecolor="white")

# Slider para cambiar slice
ax2_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
mri_slice_slider2 = Slider(ax2_slider, 'Slice', 0, 192, 100, valstep=1)

def update2(val):
    axs2.cla()
    axs2.set_title('Comparación Contornos de Máscaras')
    
    # Update slice
    mri_slice = mri_slice_slider2.val
    
    # Update Mri
    axs2.imshow(mri_image[mri_slice,:,:],cmap='gray', vmin=0, vmax=np.max(mri_image))
    
    # Update contornos
    mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice], levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linewidths=1)
    filled_mask_cocontour = axs2.contour(filled_mask[:,:,mri_slice], levels=np.logspace(-4.7, -3., 10), colors='#fe7500', linestyles='dotted', linewidths=1)
    
    # Etiquetas
    h1,_ = mask_contour.legend_elements()
    h2,_ = filled_mask_cocontour.legend_elements()

    axs2.legend([h1[0], h2[0]], ['3D U-Net', '3D U-Net Corregida'],facecolor="white")

# Actualizar plot comparación de contornos
mri_slice_slider2.on_changed(update2)