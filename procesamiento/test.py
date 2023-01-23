import numpy as np
import random
from volumentations import *
from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class gaussian_blur_3d(Transform):
    def __init__(self, sigma_range=None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.sigma_range = sigma_range
        
    def apply(self, img):
        filter_size = random.randrange(3,8,step=2) # Selecciona un tamñano de filtro (3*3*3), (5*5*5), (7*7*7), 
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1]) # Selecciona un sigma aleatorio
        return gaussian_filter(img, sigma=sigma, truncate=filter_size)

def get_augmentation(shape, intensity): 
    # Transformación de shape a aplicar
    shape_idx = random.randint(0,2) # Misma probabilidad para todas 
    shape_tran = [Rotate((-10, 10), (-10, 10), (-10, 10), p=1),
                  Flip(0, p=1),
                  RandomScale((-0.1, 0.1), p=1)]
    
    # Transformación de intensity a aplicar
    inten_idx = random.randint(0,0) # Misma probabilidad para todas 
    inten_tran = [GaussianNoise(var_limit=(10, 50), p=1),
                  gaussian_blur_3d(sigma_range=(0,0.7), p=1)] # Revisar
    
    # Lista transformaciones
    tran_list = []
    
    if shape == True:
        # Transformación flip horizontal    
        # tran_list.append(shape_tran[shape_idx])
        tran_list.append(shape_tran[2])
    
    if intensity == True:
        # tran_list.append(inten_tran[inten_idx])
        tran_list.append(inten_tran[1])
    
    return Compose(tran_list, p=1.0)
    
    # return Compose([
    #     Rotate((-10, 10), (-10, 10), (-10, 10), p=1),
    #     # RandomCropFromBorders(crop_value=0.1, p=0.5),
    #     # ElasticTransform((0, 0.25), interpolation=2, p=1),
    #     # Flip(0, p=0.5),
    #     # Flip(1, p=0.5),
    #     # Flip(2, p=0.5),
    #     # RandomRotate90((1, 2), p=0.5),
    #     # GaussianNoise(var_limit=(0, 5), p=0.2),
    #     # RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    #     # RandomScale()
    # ], p=1.0)


aug = get_augmentation(shape=False, intensity=True)

img = np.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_all_scores_split/train/MCI/S26556.npy')

# without mask
data = {'image': img}
aug_img = aug(**data)['image']

# Visualización resultados 
mri_slice = 100

# Plot Comparación máscaras
fig, axs = plt.subplots(1,2)
fig.subplots_adjust(bottom=0.15)
fig.suptitle('Comparación Máscaras Obtenidas')

axs[0].set_title('MRI original')
# axs[0].imshow(img[mri_slice,:,:],cmap='gray')
# axs[0].imshow(img[:,mri_slice,:],cmap='gray')
axs[0].imshow(img[:,:,mri_slice],cmap='gray')

axs[1].set_title('Data augmentation')
# axs[1].imshow(aug_img[mri_slice,:,:],cmap='gray')
# axs[1].imshow(aug_img[:,mri_slice,:],cmap='gray')
axs[1].imshow(aug_img[:,:,mri_slice],cmap='gray')

# Slider para cambiar slice
ax_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
mri_slice_slider = Slider(ax_slider, 'Slice', 0, 192, 100, valstep=1)

def update(val):
    mri_slice = mri_slice_slider.val
    
    axs[0].imshow(img[mri_slice,:,:],cmap='gray')
    # axs[0].imshow(img[:,mri_slice,:],cmap='gray')
    # axs[0].imshow(img[:,:,mri_slice],cmap='gray')
    
    axs[1].imshow(aug_img[mri_slice,:,:],cmap='gray')
    # axs[1].imshow(aug_img[:,mri_slice,:],cmap='gray')
    # axs[1].imshow(aug_img[:,:,mri_slice],cmap='gray')
    
# Actualizar plot comparación máscaras
mri_slice_slider.on_changed(update)