import numpy as np
import random
from volumentations import *
from scipy.ndimage import gaussian_filter
from scipy.ndimage import affine_transform

# from matplotlib import pyplot as plt
# from matplotlib.widgets import Slider

def zoom_3d(input_array, zoom_factor):
    zoom_matrix = np.array([[zoom_factor[0], 0, 0],
                            [0, zoom_factor[1], 0],
                            [0, 0, zoom_factor[2]]])
    shape = input_array.shape
    output_array = np.zeros(shape)
    output_array = affine_transform(input_array, zoom_matrix, output_shape=shape, order=1)
    return output_array

class Zoom(DualTransform):
    def __init__(self, zoom_factor_range, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.zoom_factor = np.random.uniform(zoom_factor_range[0], zoom_factor_range[1])
        self.zoom_factor_tuple = (self.zoom_factor,self.zoom_factor,self.zoom_factor)

    def apply(self, img):
        return zoom_3d(img, self.zoom_factor_tuple)

    def apply_to_mask(self, mask):
        return zoom_3d(mask, self.zoom_factor_tuple)

class GaussianBlur(Transform):
    def __init__(self, sigma_range=None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.sigma_range = sigma_range
        
    def apply(self, img):
        filter_size = random.randrange(3,8,step=2) # Selecciona un tamñano de filtro (3*3*3), (5*5*5), (7*7*7), 
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1]) # Selecciona un sigma aleatorio
        
        mask = img.copy()
        mask[mask>0] = 1
        
        return gaussian_filter(img, sigma=sigma, truncate=filter_size)*mask

class CustomGaussianNoise(Transform):
    def __init__(self, var_limit=None, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        
    def apply(self, img):
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5
        
        gaussian_noise = np.random.normal(0, sigma, img.shape)
        
        noisy_img = np.clip(img + gaussian_noise, 0, 1).astype(np.float32)
        
        mask = img.copy()
        mask[mask>0] = 1
        
        return noisy_img*mask
    
def get_augmentation(shape, intensity): 
    # Transformación de shape a aplicar
    shape_idx = random.randint(0,2) # Misma probabilidad para todas 
    shape_tran = [Rotate((-10, 10), (-10, 10), (-10, 10), p=1),
                  Flip(0, p=1),
                  Zoom((0.9, 1.1), p=1)]
    
    # Transformación de intensity a aplicar
    inten_idx = random.randint(0,1) # Misma probabilidad para todas 
    inten_tran = [CustomGaussianNoise(var_limit=(0, 0.00016), p=1), # Sigma = (0,0.4)
                  GaussianBlur(sigma_range=(0,0.7), p=1)] # Aplicar solo a cerebro
    
    # Lista transformaciones
    tran_list = []
    
    if shape == True:
        # Transformación flip horizontal    
        tran_list.append(shape_tran[shape_idx])
        # tran_list.append(shape_tran[2])
    
    if intensity == True:
        tran_list.append(inten_tran[inten_idx])
        # tran_list.append(inten_tran[1])
    
    return Compose(tran_list, p=1.0)

# aug = get_augmentation(shape=False, intensity=True)

# img = np.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Proyecto_de_Grado/Codigo/Preprocesamiento/data_all_scores_split/train/MCI/S26556.npy')

# # without mask
# data = {'image': img}
# aug_img = aug(**data)['image']

# # Visualización resultados 
# mri_slice = 100

# # Plot Comparación máscaras
# fig, axs = plt.subplots(1,2)
# fig.subplots_adjust(bottom=0.15)
# fig.suptitle('Comparación Máscaras Obtenidas')

# axs[0].set_title('MRI original')
# axs[0].imshow(img[mri_slice,:,:],cmap='gray')
# # axs[0].imshow(img[:,mri_slice,:],cmap='gray')
# # axs[0].imshow(img[:,:,mri_slice],cmap='gray')

# axs[1].set_title('Data augmentation')
# axs[1].imshow(aug_img[mri_slice,:,:],cmap='gray')
# # axs[1].imshow(aug_img[:,mri_slice,:],cmap='gray')
# # axs[1].imshow(aug_img[:,:,mri_slice],cmap='gray')

# # Slider para cambiar slice
# ax_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
# mri_slice_slider = Slider(ax_slider, 'Slice', 0, 192, 100, valstep=1)

# def update(val):
#     mri_slice = mri_slice_slider.val
    
#     axs[0].imshow(img[mri_slice,:,:],cmap='gray')
#     # axs[0].imshow(img[:,mri_slice,:],cmap='gray')
#     # axs[0].imshow(img[:,:,mri_slice],cmap='gray')
    
#     axs[1].imshow(aug_img[mri_slice,:,:],cmap='gray')
#     # axs[1].imshow(aug_img[:,mri_slice,:],cmap='gray')
#     # axs[1].imshow(aug_img[:,:,mri_slice],cmap='gray')
    
# # Actualizar plot comparación máscaras
# mri_slice_slider.on_changed(update)